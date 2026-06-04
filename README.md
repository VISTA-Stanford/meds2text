<h1 align="center">
  <a href=""><img src="assets/logo.png" alt="pipelines" width="150"></a>
  <br>
   MEDS-to-Text (<code>meds2text</code>)
  <br>
</h1>

<div align="center">

<a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-%3E%3D3.10-blue" alt="Python >= 3.10"></a> <a href="https://github.com/VISTA-Stanford/meds2text/actions/workflows/python-test.yml"><img src="https://github.com/VISTA-Stanford/meds2text/actions/workflows/python-test.yml/badge.svg?branch=main" alt="Tests"></a> <a href="https://github.com/VISTA-Stanford/meds2text/graphs/commit-activity"><img src="https://img.shields.io/badge/Maintained%3F-yes-green.svg" alt="Maintained"></a> <a href="https://github.com/psf/black"><img src="https://img.shields.io/badge/code%20style-black-000000.svg" alt="Code style: black"></a> <a href="https://github.com/VISTA-Stanford/meds2text/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License: Apache 2.0"></a>

</div>

Render an **already-transformed** MEDS parquet extract into text representations.

> [!IMPORTANT]
> `meds2text` does **not** transform/clean data. It assumes the input is a MEDS
> parquet extract (`data/**/*.parquet`) that has already been cleaned upstream.
> All OMOP/STARR transforms (orphan repair, visit-interval adjustment, billing
> code moves, ICD10 → ICD10CM, delta encoding, flowsheet handling, etc.) now live
> in [`medspace`](https://github.com/VISTA-Stanford/medspace); run e.g.
> `medspace transform --preset STARR_CLEAN` before textifying.

> [!NOTE]
> Flowsheet enrichment (flattening `STANFORD_OBS/Flowsheet` JSON into
> `name` / `unit_source_value` / `group_name`) is intentionally **not** handled
> here. `medspace.transforms.parse_flowsheet_json` currently only extracts the
> measurement value; if richer flowsheet rendering is needed, extend it there.

### Projects Using `meds2text`

- [Medalign: A clinician-generated dataset for instruction following with electronic medical records](https://ojs.aaai.org/index.php/AAAI/article/view/30205) (AAAI 2024)
- [TIMER: Temporal Instruction Modeling and Evaluation for Longitudinal Clinical Records](https://arxiv.org/abs/2503.04176) (npj Digital Medicine 2025)

## 🚀 Installation

For development (from source):

```bash
git clone https://github.com/VISTA-Stanford/meds2text
cd meds2text
pip install -e .[dev]
```

## 📦 Data Dependencies

- **Athena Vocabularies**: Required for OMOP concept mapping. See section "I. Athena Vocabularies" for guidance on where to download or build from scratch.
- **Metadata (OPTIONAL)**: Events can be linked to `care_site_id`, `provider_id`, and `payer_plan` via external dataframes if not present in the MEDS extract.
- **MEDS Extract**: Can be generated using internal STARR OMOP CDM data or via existing public extracts.
  - [MedAlign](https://stanford.redivis.com/datasets/48nr-frxd97exb)
  - [INSPECT](https://stanford.redivis.com/datasets/dzc6-9jyt6gapt)
  - [EHRSHOT](https://stanford.redivis.com/datasets/53gc-8rhx41kgt)
  - [MIMIC-IV](https://physionet.org/content/mimiciv/)

## ⚡ Quick Start: Textifying a MEDS Extract

Convert INSPECT (or any MEDS dataset) to LUMIA XML markup. This should take ~10 minutes for 19,391 patients using 8 CPU cores.

### Default: Export All Structured Data + Notes

```bash
meds-textify \
--path_to_meds data/meds_extracts/omop_inspect/ \
--path_to_ontology data/athena_omop_ontologies/ \
--path_to_metadata data/omop_metadata/ \
--path_to_output data/inspect_lumia_xml/ \
--exclude_props clarity_table \
--format lumia_xml \
--include_contexts person providers care_sites \
--event_types "*" \
--n_processes 8 
```

### Options: Format

Set with `--format`:

- `lumia_xml` \[default\] (see [specification](docs/markup.md))
- `lumia_json`
- `fhir_like_json` (sketch only)

By default one file is written per subject. Pass `--batch_mode` (with `--batch_size`) to instead write batched JSONL files (`batch_<n>.jsonl`), one record per subject.

### Options: Filter Properties

Events have a number of properties defined by the source ETL. Some properties are useful for materializing a graph view of event provenance, e.g., `clarity_table`, `provider_id`, `visit_id`, `care_site_id`. Other properties make sense for OMOP CDM sourced data, which provides standardzied event `type` properties. 

To miminize markup bloat, you can filter out events using `--exclude_props`.

### Options: Filter Events

Generate markup for **notes + visits** and exclude providers + care_sites

```bash
meds-textify \
--path_to_meds data/meds_extracts/omop_inspect/ \
--path_to_ontology data/athena_omop_ontologies/ \
--path_to_metadata data/omop_metadata/ \
--path_to_output data/inspect_lumia_xml/ \
--exclude_props clarity_table visit_id provider_id \
--format lumia_xml \
--include_contexts person \
--event_types note visit_detail visit \
--n_processes 8
```

Generate markup for **only structured data** and exclude providers + care_sites


```bash
meds-textify \
--path_to_meds data/meds_extracts/omop_inspect/ \
--path_to_ontology data/athena_omop_ontologies/ \
--path_to_metadata data/omop_metadata/ \
--path_to_output data/inspect_lumia_xml/ \
--exclude_props clarity_table visit_id provider_id \
--format lumia_xml \
--include_contexts person \
--event_types condition death device_exposure drug_exposure image measurement observation procedure visit visit_detail \
--n_processes 8
```

You can also drop events by code pattern (wildcards supported) with `--exclude_codes`, e.g. `--exclude_codes "STANFORD_OBS/*"`.

### Options: Person Context

When `--include_contexts person` is set, a `<person>` block is attached to each encounter. You control which sub-blocks appear, and where, with two flags:

- `--person_fields_every_encounter` (default: `age payerplan`) — fields that change over time and are repeated in every encounter.
- `--person_fields_first_encounter` (default: `birthdate demographics`) — static fields emitted only in the first encounter.

Valid fields are `birthdate`, `age`, `demographics`, and `payerplan`. A field listed in neither flag is omitted entirely. For example, to repeat the birthdate in every encounter and drop demographics:

```bash
meds-textify \
--path_to_meds data/meds_extracts/omop_inspect/ \
--path_to_ontology data/athena_omop_ontologies/ \
--path_to_metadata data/omop_metadata/ \
--path_to_output data/inspect_lumia_xml/ \
--include_contexts person \
--person_fields_every_encounter age payerplan birthdate \
--person_fields_first_encounter
```

### Options: Other

- `--person_ids_file`: CSV/TSV with a `person_id` column to restrict processing to a subset of subjects.
- `--attribute_order`: preferred ordering of XML attributes (default: `table code name`).
- `--batch_mode` / `--batch_size`: write batched JSONL instead of one file per subject.
- `--test_mode`: process only a handful of subjects for quick smoke tests.

## 🛠️ Detailed Reproduction Steps

For mapping codes to text strings, we rely on medical vocabularies and ontologies. For MEDS extracted souced from OMOP CDM datasets, we can use a pre-packed collection of vocabularies.

### I. Athena Vocabularies

The OMOP vocabulary is made available through the [Athena – OHDSI Vocabularies Repository](https://athena.ohdsi.org/). The majority of vocabularies can be redistributed for academic/non-commerical purposes, with the exception of CPT4. 

#### ☁️ Option A. Download Prebuilt Dependencies from GCS

If you have access to the Stanford VISTA project, you can download all prebuilt dependencies from:

```bash
gsutil -m cp -r gs://su-vista/shah_lab/meds2text/data ./data
```

#### 🏗️ Option B. Create from Scratch

**1. Download Public Vocabularies**

You can do either (1) download a [cached snapshot here](https://drive.google.com/drive/folders/1F59yqlGzyYOWQoa7nHQJSRc1gpl6Lo--?usp=sharing) OR (2) create an account and login to Athena to [download the most recent version of public vocabularies](https://athena.ohdsi.org/vocabulary/list).

**2. Download and Update with the CPT4 Vocabulary** 

- CPT4 cannot be redistributed and must be added manually. Generate an API key from your [UMLS account profile](https://uts.nlm.nih.gov/uts.html#profile) and follow the Athena instructions (the bundled `cpt.sh` script in your Athena download) to reconstitute CPT4 into the vocabulary files.
- Place the updated vocabulary folder in your `data/` directory (e.g., `data/athena_ontologies_snapshot/`).

**3. Create Dataframes and Prefix Trie** 

Run the following script to initialize the parquet and trie files for fast ontology lookup.

```bash
python scripts/init_athena_ontologies.py \
--athena_path data/athena_ontologies_snapshot.zip \
--custom_mappings data/stanford_custom_concepts.csv.gz \
--save_parquet data/athena_omop_ontologies
```

### II. MEDS Extracts

See [here](https://github.com/VISTA-Stanford/ehr-tumorboard?tab=readme-ov-file#examples) for working instructions on generating STARR MEDS extracts.

## 📄 License

Apache 2.0 License. See [LICENSE](LICENSE) for details.


