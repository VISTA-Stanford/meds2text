"""Clinical-note text compression for LUMIA XML output.

Conservative, mostly lossless transforms wired behind ``--compress_notes``:

1. **De-identification scrub** — remove masked IDs / placeholder phones (lossless
   for clinical meaning).
2. **Unicode whitespace normalize** — map NBSP and other unicode space chars to
   ASCII space (lossless for token content).
3. **Repeated-symbol collapse** — decoration runs like ``********`` or long
   ``----`` / ``====`` separators collapse to a single character (lossless for
   clinical meaning).
4. **Whitespace collapse** — normalize runs of whitespace to a single space
   (lossless for token content; ``str.split()`` is unchanged).
5. **PMC 10-gram dedup** — drop tokens that repeat a 10-token window seen in an
   earlier note or earlier in the same note (intentionally lossy: removes redundant
   copy-paste, not unique clinical facts).

Notes are processed in global chronological order so cross-note dedup is correct.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Set, Tuple

from meds2text.subject import Event, Subject

_DEID_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\(?\b999[-.]?999[-.]?9999\b\)?"),
    re.compile(r"\b9999-9999\b"),
    re.compile(r"MRN:?\s*\[0+\]"),
    re.compile(r"CSN:?\s*\[0+\]"),
    re.compile(r"\[0{4,}\]"),
    re.compile(r"@[A-Z]+@"),
)
_MULTI_SPACE = re.compile(r"  +")
_RE_NOTE_WS_RUNS = re.compile(r"\s+")
# Decoration/separator chars: collapse 5+ identical runs to one (``----`` kept).
_RE_REPEATED_SYMBOLS = re.compile(r"([*\-_=.+#|~\\])\1{4,}")
# Common unicode space codepoints → ASCII space (before general ws collapse).
_UNICODE_SPACES = dict.fromkeys(
    (
        "\u00a0",  # NBSP
        "\u2000",
        "\u2001",
        "\u2002",
        "\u2003",
        "\u2004",
        "\u2005",
        "\u2006",
        "\u2007",
        "\u2008",
        "\u2009",
        "\u200a",
        "\u202f",
        "\u205f",
        "\u3000",
    ),
    " ",
)

DEFAULT_NGRAM_SIZE = 10


def scrub_deid(text: str) -> str:
    """Remove common de-identification artifacts from note prose."""
    for pat in _DEID_PATTERNS:
        text = pat.sub("", text)
    return _MULTI_SPACE.sub(" ", text)


def normalize_unicode_whitespace(text: str) -> str:
    """Replace common unicode space characters with ASCII space."""
    if not any(ch in text for ch in _UNICODE_SPACES):
        return text
    for src, dst in _UNICODE_SPACES.items():
        text = text.replace(src, dst)
    return text


def collapse_repeated_symbols(text: str) -> str:
    """Collapse long runs of decoration/separator characters to a single char."""
    return _RE_REPEATED_SYMBOLS.sub(r"\1", text)


def collapse_whitespace(text: str) -> str:
    """Strip ends and collapse internal whitespace runs to one ASCII space."""
    return _RE_NOTE_WS_RUNS.sub(" ", text.strip())


def mark_duplicate_token_indices(
    tokens: List[str],
    n: int,
    prior_notes_ngrams: Set[Tuple[str, ...]],
) -> Set[int]:
    """Mark token indices that belong to a duplicate *n*-gram window."""
    redundant: Set[int] = set()
    length = len(tokens)
    if length < n:
        return redundant
    seen_in_this_note: Set[Tuple[str, ...]] = set()
    for i in range(0, length - n + 1):
        gram = tuple(tokens[i : i + n])
        if gram in prior_notes_ngrams or gram in seen_in_this_note:
            for j in range(i, i + n):
                redundant.add(j)
        seen_in_this_note.add(gram)
    return redundant


def add_all_ngrams(tokens: List[str], n: int, target: Set[Tuple[str, ...]]) -> None:
    """Register every *n*-gram from ``tokens`` into ``target``."""
    if len(tokens) < n:
        return
    for i in range(0, len(tokens) - n + 1):
        target.add(tuple(tokens[i : i + n]))


def dedup_ngrams(
    text: str,
    n: int,
    prior_notes_ngrams: Set[Tuple[str, ...]],
) -> str:
    """Drop tokens participating in duplicate *n*-grams; update ``prior`` set."""
    if not text.strip():
        return text
    tokens = text.split()
    if len(tokens) < n:
        add_all_ngrams(tokens, n, prior_notes_ngrams)
        return text
    redundant = mark_duplicate_token_indices(tokens, n, prior_notes_ngrams)
    out_tokens = [tokens[i] for i in range(len(tokens)) if i not in redundant]
    add_all_ngrams(tokens, n, prior_notes_ngrams)
    return " ".join(out_tokens)


@dataclass
class NoteCompressor:
    """Stateful compressor: chronological 10-gram dedup across a subject's notes."""

    scrub_deid_enabled: bool = True
    normalize_unicode_whitespace_enabled: bool = True
    collapse_repeated_symbols_enabled: bool = True
    collapse_whitespace_enabled: bool = True
    dedup_ngrams_enabled: bool = True
    ngram_size: int = DEFAULT_NGRAM_SIZE
    _prior_ngrams: Set[Tuple[str, ...]] = field(default_factory=set)

    def compress(self, text: str) -> str:
        if not text or not text.strip():
            return text
        out = text
        if self.scrub_deid_enabled:
            out = scrub_deid(out)
        if self.normalize_unicode_whitespace_enabled:
            out = normalize_unicode_whitespace(out)
        if self.collapse_repeated_symbols_enabled:
            out = collapse_repeated_symbols(out)
        if self.collapse_whitespace_enabled:
            out = collapse_whitespace(out)
        if self.dedup_ngrams_enabled:
            out = dedup_ngrams(out, self.ngram_size, self._prior_ngrams)
        return out


def _is_note_event(event: Event) -> bool:
    return getattr(event, "table", None) == "note"


def _note_sort_key(event: Event) -> tuple:
    t = getattr(event, "time", None)
    return (t is None, t or datetime.min, str(event.code))


def compress_subject_notes(subject: Subject) -> None:
    """Compress note ``text_value`` on ``subject`` in chronological order (in place)."""
    notes = [e for e in subject.events if _is_note_event(e)]
    notes.sort(key=_note_sort_key)
    compressor = NoteCompressor()
    for event in notes:
        raw = getattr(event, "text_value", None)
        if raw is None or not str(raw).strip():
            continue
        compressed = compressor.compress(str(raw))
        event.properties["text_value"] = compressed
