""" 

Usage Example:


python apps/timeline_viewer/app.py \
--timeline_dir data/collections/dev-corpus/ \
--person_id 135918373

"""

import os
import re
import html
import argparse
import gradio as gr
import xml.etree.ElementTree as ET
from datetime import datetime


# ── 0. Command-line args ──
parser = argparse.ArgumentParser(description="Patient Timeline Viewer")
parser.add_argument(
    "--timeline_dir",
    "-d",
    help="Directory containing timeline XML files (default: current dir)",
)
parser.add_argument(
    "--person_id",
    "-p",
    help="Unique substring of XML filename to load on startup",
)
args = parser.parse_args()

initial_root_dir = args.timeline_dir or "."
initial_person_id = args.person_id or ""

# ── 1. Helpers for XML parsing & categorization ──


def categorize_encounter(enc):
    """
    Determine encounter category based on visit and visit_detail text.
    """
    visit = enc.findtext("visit", "").lower()
    visit_detail = enc.findtext("visit_detail", "").lower()
    if any(
        keyword in visit_detail for keyword in ("inpatient", "admission", "hospital")
    ):
        return "inpatient"
    elif any(keyword in visit for keyword in ("outpatient", "clinic")):
        return "outpatient"
    elif any(keyword in visit for keyword in ("phone", "telephone")):
        return "telephone"
    else:
        return "default"


def parse_xml_to_data(xml_string):
    root = ET.fromstring(xml_string)
    encounters = []
    for idx, enc in enumerate(root.findall("encounter")):
        age = enc.findtext("person/age/years", "?")
        gender = enc.findtext("person/demographics/gender", "?").capitalize()
        ethnicity = enc.findtext("person/demographics/ethnicity", "?").capitalize()
        enc_cat = categorize_encounter(enc)

        entries = enc.find("events").findall("entry")
        events_list, timestamps = [], []
        for entry in entries:
            ts = entry.get("timestamp")
            if ts:
                timestamps.append(ts)
            evs = []
            for e in entry.findall("event"):

                # —— HACK: for image events, override name attribute
                if e.get("type") == "image":
                    site = e.get("anatomic_site_source_value", "")
                    mod = e.get("modality_source_value", "")
                    # build "CHEST CR" or " SR", then strip leading/trailing spaces
                    e.set("name", f"{site} {mod}".strip())
                # -------------------------------------------------------

                t = e.get("type", "default")
                evs.append(
                    {
                        "type": t,
                        "name": e.get("name", "Unknown"),
                        "value": (e.text or "").strip(),
                        "timestamp": ts,
                        "code": e.get("code", ""),
                    }
                )
            if evs:
                events_list.append({"timestamp": ts, "events": evs})

        if timestamps:
            dates = [datetime.fromisoformat(t) for t in timestamps]
            start, end = min(dates), max(dates)
            start_str = start.date().isoformat()
            duration = (end - start).days + 1
        else:
            start_str, duration = "?", "?"

        encounters.append(
            {
                "index": idx + 1,
                "age": age,
                "gender": gender,
                "ethnicity": ethnicity,
                "start_date": start_str,
                "duration_days": duration,
                "events": events_list,
                "event_count": sum(len(x["events"]) for x in events_list),
                "enc_category": enc_cat,
            }
        )
    return encounters


# ── 2. Icon & label mappings ──

ICON_MAP = {
    "visit": "🏥",
    "visit_detail": "🛏️",
    "drug_exposure": "💊",
    "measurement": "📏",
    "procedure": "🛠️",
    "observation": "🔍",
    "note": "📝",
    "image": "🩻",
    "condition": "⚕️",
    "device_exposure": "🔩",
    "death": "🌸",
    "default": "•",
}
LABEL_MAP = {k: f"{ICON_MAP.get(k,'•')} {k}" for k in ICON_MAP if k != "default"}
LABEL_TO_TYPE = {v: k for k, v in LABEL_MAP.items()}

ENCOUNTER_ICON = {
    "inpatient": "🛏️",
    "outpatient": "🏥",
    "telephone": "📞",
    "default": "👤",
}
DEFAULT_BG = "#eee"

# ── 3. Highlight & heatmap coloring ──


def highlight(text, q):
    if not q:
        return text  # Don't escape HTML tags
    try:
        # Only escape the search query, not the entire text
        esc_q = html.escape(q)
        return re.sub(f"({esc_q})", r"<mark>\1</mark>", text, flags=re.IGNORECASE)
    except re.error:
        return text


def compute_event_density_color(count, lo, hi):
    norm = 0 if hi == lo else (count - lo) / (hi - lo)
    light = 90 - int(norm * 50)
    return f"hsl(200,80%,{light}%)"


# ── 4. HTML rendering ──


def render_html(
    encs, emoji_labels, heatmap, search_q, expand_all, truncate_notes, scope, invert
):
    encs_to_render = list(reversed(encs)) if invert else encs
    counts = [e["event_count"] for e in encs_to_render]
    lo, hi = min(counts), max(counts)
    try:
        pat = re.compile(search_q, re.IGNORECASE)
    except re.error:
        pat = None

    html_out = ""
    for e in encs_to_render:
        types_in = [LABEL_TO_TYPE[l] for l in emoji_labels]
        filtered = []
        for entry in e["events"]:
            evs = []
            for ev in entry["events"]:
                if ev["type"] not in types_in:
                    continue

                # Search in raw event data
                search_matches = False
                if search_q:
                    try:
                        pat = re.compile(search_q, re.IGNORECASE)
                        search_matches = (
                            (scope in ("name", "both") and pat.search(ev["name"]))
                            or (scope in ("value", "both") and pat.search(ev["value"]))
                            or (
                                scope in ("code", "both")
                                and ev.get("code")
                                and pat.search(ev["code"])
                            )
                        )
                    except re.error:
                        search_matches = False
                else:
                    search_matches = True

                if not search_matches:
                    continue

                evs.append(ev)
            if evs:
                filtered.append({"timestamp": entry["timestamp"], "events": evs})
        if not filtered:
            continue
        if invert:
            filtered = list(reversed(filtered))
        bg = (
            compute_event_density_color(e["event_count"], lo, hi)
            if heatmap
            else DEFAULT_BG
        )
        enc_icon = ENCOUNTER_ICON.get(
            e.get("enc_category", "default"), ENCOUNTER_ICON["default"]
        )
        matched_count = sum(len(ent["events"]) for ent in filtered)
        event_count_text = f" — {e['event_count']} events" + (
            f" (matched {matched_count} events)" if search_q else ""
        )
        summary = (
            f"{enc_icon} <b>Encounter {e['index']}</b>"
            f" (Start: {e['start_date']} • Duration: {e['duration_days']}d • "
            f"Age {e['age']}, {e['gender']}, {e['ethnicity']}){event_count_text}"
        )
        body = ""
        for ent in filtered:
            body += f"<details {'open' if expand_all else ''}>"
            body += (
                f"<summary>{ent['timestamp']}</summary><div style='margin-left:10px;'>"
            )
            for ev in ent["events"]:
                icon = ICON_MAP.get(ev["type"], "•")
                nm = highlight(ev["name"], search_q)
                raw = ev["value"] or ""

                # Preserve whitespace and handle truncation for notes
                if ev["type"] == "note":
                    # Normalize multiple spaces to single <br>
                    raw = re.sub(r"\s{2,}", "<br>", raw)

                    if truncate_notes and len(raw) > 250:
                        # Find a good breaking point near 250 chars
                        break_point = raw[:250].rfind("<br>")
                        if break_point > 0:
                            raw = raw[:break_point] + "..."
                        else:
                            break_point = raw[:250].rfind(" ")
                            if break_point > 0:
                                raw = raw[:break_point] + "..."
                            else:
                                raw = raw[:250] + "..."
                    # Wrap in div with border and padding
                    raw = f"<div style='border:1px solid #ADADAD;padding:8px;margin:4px 0;'>{raw}</div>"

                val_h = highlight(raw, search_q)
                low = raw.strip().lower()
                if low == "start":
                    val = f"🟢 {val_h}"
                elif low == "stop" or low == "end":
                    val = f"🔴 {val_h}"
                elif low == "start|end":
                    val = f"‼️ {val_h}"
                else:
                    val = val_h
                code = ev.get("code", "")

                # Create the code display span
                highlighted_code = highlight(code, search_q) if search_q else code
                code_display = (
                    f" <span style='display:inline-block;border:1px solid #ccc;border-radius:4px;padding:1px 6px;font-family:monospace;font-weight:bold;font-size:0.85em;'>{highlighted_code}</span>"
                    if code
                    else ""
                )

                # Highlight each part separately
                if search_q:
                    icon = highlight(icon, search_q)
                    nm = highlight(nm, search_q)
                    val = highlight(val, search_q)

                body += f"<div>{icon} <b>{nm}</b>{code_display}: {val}</div>"
            body += "</div></details>"
        html_out += (
            f"<details {'open' if expand_all else ''}>"
            f"<summary style='background-color:{bg};padding:6px;border-radius:4px;'>{summary}</summary>"
            f"<div style='margin-left:15px;'>{body}</div></details>"
        )

    return (
        "<div style='max-height:600px;overflow-y:auto;padding:10px;"
        "border:1px solid #ccc;font-size:16px;line-height:1.5;font-family:sans-serif;'>"
        + html_out
        + "</div>"
    )


# ── 5. Directory scanning & load function ──


def find_xml_file(root_dir: str, name_filter: str):
    """
    Returns the first XML file under root_dir whose filename contains name_filter.
    """
    try:
        for entry in os.scandir(root_dir):
            if (
                entry.is_file()
                and entry.name.endswith(".xml")
                and name_filter in entry.name
            ):
                return entry.path
    except Exception:
        pass
    return None


def load_file_and_render(
    root_dir,
    file_filter,
    emoji_labels,
    heatmap,
    search_q,
    expand_all,
    truncate_notes,
    scope,
    invert,
):
    xml_path = find_xml_file(root_dir.strip(), file_filter.strip())
    if xml_path is None:
        return (
            "<div style='color:red;padding:10px;'>"
            f"No XML found matching '{file_filter}' in {root_dir}"
            "</div>"
        )
    with open(xml_path, encoding="utf-8") as f:
        xml_string = f.read().lstrip()
    data = parse_xml_to_data(xml_string)
    return render_html(
        data, emoji_labels, heatmap, search_q, expand_all, truncate_notes, scope, invert
    )


def reset_event_types():
    return list(LABEL_MAP.values())


# ── 6. Gradio app ──

emoji_choices = list(LABEL_MAP.values())

with gr.Blocks() as demo:
    gr.Markdown("## 👤 Patient Timeline Viewer")

    # Directory chooser + filter
    with gr.Row():
        root_dir_tb = gr.Textbox(
            label="Timeline Directory", value=initial_root_dir, interactive=True
        )
        file_filter_tb = gr.Textbox(
            label="person_id",
            value=initial_person_id,
            placeholder="type a unique part of the filename",
            interactive=True,
        )
        load_btn = gr.Button("Load File")

    # Event-type & display controls
    with gr.Row():
        with gr.Column(scale=2):
            event_types = gr.CheckboxGroup(
                choices=emoji_choices,
                value=emoji_choices,
                label="Event Types",
                interactive=True,
            )
            reset_btn = gr.Button("Reset")
        with gr.Column(scale=1):
            heat_cb = gr.Checkbox(label="Enable Heatmap", value=True)
            trunc_cb = gr.Checkbox(label="Truncate Notes", value=False)
            scope_dd = gr.Dropdown(
                choices=["name", "value", "both"], value="both", label="Search Scope"
            )

    with gr.Row():
        with gr.Column(scale=8):
            search_in = gr.Textbox(
                label="Search (Regex OK)",
                placeholder="e.g., glucose|follow-follow",
                interactive=True,
            )
        with gr.Column(scale=1):
            expand_cb = gr.Checkbox(label="Expand All", value=False, interactive=True)
            invert_cb = gr.Checkbox(
                label="Invert Timeline", value=True, interactive=True
            )

    timeline_out = gr.HTML()

    # Wire up interactions
    reset_btn.click(fn=reset_event_types, inputs=None, outputs=event_types)

    load_btn.click(
        fn=load_file_and_render,
        inputs=[
            root_dir_tb,
            file_filter_tb,
            event_types,
            heat_cb,
            search_in,
            expand_cb,
            trunc_cb,
            scope_dd,
            invert_cb,
        ],
        outputs=timeline_out,
    )

    # Allow tweaking filters after load
    controls = [
        root_dir_tb,
        file_filter_tb,
        event_types,
        heat_cb,
        search_in,
        expand_cb,
        trunc_cb,
        scope_dd,
        invert_cb,
    ]
    for c in controls[2:]:
        c.change(fn=load_file_and_render, inputs=controls, outputs=timeline_out)

    # If both command-line args were provided, auto‐load on startup:
    if args.timeline_dir and args.person_id:
        demo.load(
            fn=load_file_and_render,
            inputs=[
                root_dir_tb,
                file_filter_tb,
                event_types,
                heat_cb,
                search_in,
                expand_cb,
                trunc_cb,
                scope_dd,
                invert_cb,
            ],
            outputs=timeline_out,
        )

    demo.launch()
