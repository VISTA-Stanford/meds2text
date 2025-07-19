# EHR Text Linearization

## **LUMIA XML Schema Documentation for Event Streams**

   This document outlines the structure and elements of the XML Schema used to describe patient encounter and event data in a healthcare setting. The schema captures details about the patient, encounters with healthcare providers, and various related events.

**Root Element**

- **`<eventstream>`**
  - **Attributes:**
    - `person_id` (string): Unique identifier for the person whose healthcare events are being recorded.
  - **Child Elements:**
    - `<encounter>`: Represents a specific encounter in which events occurred pertaining to the patient.

**Encounter Element**

- **`<encounter>`**
  - **Attributes:**
    - `start_timestamp` (datetime): The starting date and time of the encounter.
    - `end_timestamp` (datetime): The ending date and time of the encounter.
  - **Child Elements:**
    - `<person>`: Contains demographic and personal information about the patient.
    - `<caresites>`: Details the care sites where the encounter occurred.
    - `<providers>`: Information regarding the healthcare providers involved in the encounter.
    - `<events>`: A collection of events that took place during this encounter.

- **`<person>`**
  - **Child Elements:**
    - `<birthdate>` (date): Birth date of the person.
    - `<age>`: Contains the age of the person.
      - `<days>` (integer): Age in days.
      - `<years>` (integer): Age in years.
    - `<demographics>`: Information about the demographics of the patient.
      - `<ethnicity>` (string): Ethnic background of the patient.
      - `<gender>` (string): Gender of the patient.
    - `<payerplan>` (string): Information about the patient's insurance plan (currently can be empty).

- **`<caresites>`**
  - **Child Elements:**
    - `<caresite>`: Represents a site where care was provided.
      - **Attributes:**
        - `care_site_id` (string): Unique identifier for the care site.
        - `care_site_name` (string): Name of the care site.

- **`<providers>`**
  - **Child Elements:**
    - `<provider>`: Represents a healthcare provider involved in the encounter.
      - **Attributes:**
        - `provider_id` (string): Unique identifier for the provider.
        - `gender` (string): Gender of the provider (may be "NULL" if not available).
        - `speciality` (string): Provider's specialty field (may be "NULL" if not available).
        - `year_of_birth` (integer): Year of birth of the provider (may be "NULL" if not available).
        - `care_site_name` (string): Name of the care site where the provider practices (may be "NULL" if not available).

- **`<events>`**
  - **Child Elements:**
    - `<entry>`: An entry representing an event that occurred during the encounter.
      - **Attributes:**
        - `timestamp` (datetime): When the event occurred.
      - **Child Elements:**
        - `<event>`: Describes a specific event that occurred.
          - **Attributes:**
            - `note_id` (string, optional): Unique identifier for the note (may be empty).
            - `provider_id` (string): Unique identifier for the provider related to the event.
            - `care_site_id` (string, optional): Identifier for the care site where the event occurred (may be empty).
            - `type` (string): Type of event (e.g., "note", "visit", "visit_detail").
            - `code` (string): Code representing the event type in standard coding systems (e.g., LOINC, NUCC).
            - `name` (string): Name or description of the event. Can be empty for certain events.

**Example Structure**

```xml
<eventstream person_id="2129690">
  <encounter start_timestamp="2013-10-10 19:08:00" end_timestamp="2018-12-07 23:59:00">
    <person>
      <birthdate>1970-05-01</birthdate>
      <age>
        <days>15838</days>
        <years>43</years>
      </age>
      <demographics>
        <ethnicity>Not Hispanic or Latino</ethnicity>
        <gender>FEMALE</gender>
      </demographics>
      <payerplan>Medicare</payerplan>
    </person>
    ...
  </encounter>
</eventstream>
```

