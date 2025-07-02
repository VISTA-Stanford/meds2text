# MEDS to Text 
Transform MEDS-formatted data into text representations. 

> [!WARNING]  
> Currently `meds2text` assumes the MEDS extract is sourced from OMOP CDM sources.

This repo requires a MEDS dataset. These can be created from MIMIC (see XXX) or our 
internal STARR OMOP CDM datasets.  


## Data Dependencies 

- **Athena Vocabularies**: This is cumbersome because CPT4 codes cannot be redistributed by users and must be downloaded from the source vendor. 

- **Metadata (OPTIONAL)**: Events are linkable to `care_site_id` and `provider_id` and encounters are assocaited with a `payer_plan`. These attributes can be encoded in the MEDS extract, however for now we use a hack to provide them via external dataframes. 

## Usage Example

