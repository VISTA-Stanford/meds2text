# Anonymized excerpts preserving structure from LUMIA / medalign-style note text
# (used to test table-like block stripping).

PROCEDURE_NOTE_ASCII = (
    "The following orders were created for panel order CBC WITH DIFFERENTIAL. "
    "Procedure                               Abnormality         Status                    "
    "---------                               -----------         ------                    "
    "CBC WITH DIFFERENTIAL[[000000]]        Abnormal            Final result              "
    "MANUAL DIFFERENTIAL/SLID...[[000000]]  Abnormal            Final result              "
    "Please view results for these tests on the individual orders."
)

RECENT_RESULTS_EMBEDDED = (
    "Labs:  Recent Results (from the past 24 hour(s))   Istat CG4, Venous    "
    "Collection Time: 01/25/2017   Result Value Ref Range    "
    "PH (v), ISTAT 7.40 7.32 - 7.42    PCO2 (v), ISTAT 38.9 (L) 40.0 - 50.0 mmHg    "
    "I have reviewed the labs.     Echo:   Prelim read: normal."
)

RECENT_RESULTS_TO_RADIOLOGY = (
    "Recent Results (from the past 48 hour(s))   Metabolic Panel    "
    "Collection Time: 02/01/2017   Result Value Ref Range    Sodium 140 130 - 145 mmol/L    "
    "Radiology & Imaging Studies: (completed)  None"
)

ECG_NOTE = (
    "-------------------- Pediatric ECG interpretation -------------------- "
    "FINDINGS:                 Sinus rhythm                 HEART RATE = 135 bpm                 "
    "INTERPRETATION: - ABNORMAL ECG - Signed."
)

LINES_DRAINS_TO_VITALS = (
    "Line Access Information:    Patient Lines/Drains/Airways Status    "
    "Active Lines     None                        | Vital Signs: Temp: 36.6"
)

PIPE_METADATA_HEADER = (
    "NEONATAL PROGRESS NOTE      NAME: Patient A  MRN: [000000] DOB: 01/23/2017 LOC: NICU270  "
    "Admitting Service: Neonatology | Admission Date: 01/25/2017 | Hospital Day: 2 | "
    "Day of Life: 4 | Unknown | 40w1d    Date of Service: 01/26/2017     Identification: stable"
)
