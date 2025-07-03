# MEDS to Text
Transform MEDS-formatted data into text representations from OMOP CDM sources.

> [!WARNING]
> Currently `meds2text` assumes the MEDS extract is sourced from OMOP CDM sources.
> This repo is not optimized for effeciency. 

## 🚀 Installation

For development (from source):

```bash
git clone https://github.com/VISTA-Stanford/meds2text
cd meds2text
pip install -e .[dev]
```

## 📦 Data Dependencies

- **Athena Vocabularies**: Required for OMOP concept mapping. CPT4 codes must be downloaded directly from the source vendor due to licensing restrictions.
- **Metadata (OPTIONAL)**: Events can be linked to `care_site_id`, `provider_id`, and `payer_plan` via external dataframes if not present in the MEDS extract.
- **MEDS Extract**: Can be generated using internal STARR OMOP CDM data or via existing public extracts.
 - [MedAlign](https://stanford.redivis.com/datasets/48nr-frxd97exb)
 - [INSPECT](https://stanford.redivis.com/datasets/dzc6-9jyt6gapt)
 - [EHRSHOT](https://stanford.redivis.com/datasets/53gc-8rhx41kgt)
 - MIMIC-IV 	

## ⚡ Quick Start: Textifying a MEDS Extract

Convert INSPECT (or any MEDS dataset) to LUMIA XML markup. This tasks ~10 minutes for 19,391 patients using 10 CPU cores.

```bash
python src/meds2text/textify.py \
--path_to_meds meds_reader_omop_inspect/ \
--path_to_ontology data/athena_omop_ontologies/ \
--path_to_metadata data/omop_metadata/ \
--path_to_output data/medalign_lumia_xml/ \
--exclude_props clarity_table \
--format lumia_xml \
--include_contexts person providers care_sites \
--apply_transforms \
--n_processes 10 
```

## 🛠️ Detailed Reproduction Steps

For mapping codes to text strings, we rely on medical vocabularies and ontologies. For MEDS extracted souced from OMOP CDM datasets, we can use a pre-packed collection of vocabularies.

### ☁️ Option A. Download Prebuilt Dependencies from GCS

If you have access to the Stanford VISTA project, you can download all dependencies from:

```bash
gsutil -m cp -r gs://su-vista/shah_lab/meds2text/data ./data
```

### 🏗️ Option B. Create from Scratch

- Register and download the vocabularies from [OHDSI Athena](https://athena.ohdsi.org/vocabulary/list).
- Generate an API KEY for your [UMLS account profile](https://uts.nlm.nih.gov/uts.html#profile) and use `cpt.sh` to download the CPT4 vocabulary and auto-update Athena vocabularies
- Place the resulting folder (or zip file) in your `data/` directory (e.g., `data/athena_ontologies_snapshot/`).

Run the following script to initialize the parquet and trie files for fast ontology lookup.

```bash
python scripts/init_athena_ontologies.py \
--athena_path data/athena_ontologies_snapshot.zip \
--custom_mappings data/stanford_custom_concepts.csv.gz \
--save_parquet data/athena_omop_ontologies
```

## 📄 License

MIT License. See [LICENSE](LICENSE) for details.


