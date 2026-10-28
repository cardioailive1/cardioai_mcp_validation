"""Inventory the MCP workspace: datasets, tables, columns WITH their types, care sites, note titles.

MCP provisions the data in BigQuery (2026-10-28). Run once at onboarding, then fill config/study.yaml (schemas,
site_map) and config/modality_tables.yaml (the CONFIRM column names):
    python scripts/discover_schema.py --out exports/schema_inventory
    python scripts/discover_schema.py --project <project id> --out exports/schema_inventory

Writes names, types and counts only (no patient rows), so the files can be shared to fill in the config.
Column types matter: BigQuery does not compare DATETIME with TIMESTAMP, or STRING with INT64, without a cast.
"""
from __future__ import annotations

import argparse
import os
import re

import pandas as pd

OMOP = ["person", "observation_period", "visit_occurrence", "condition_occurrence", "procedure_occurrence", "drug_exposure",
        "measurement", "observation", "device_exposure", "death", "note", "note_nlp", "care_site", "location",
        "concept", "concept_relationship", "concept_ancestor"]
MODALITY = re.compile(r"(ecg|ekg|waveform|wave|echo|dicom|image|imaging|study|series|cine|radiology|patient)", re.I)
# columns the repo's config still marks CONFIRM: printed so they can be checked at a glance
WATCH = ["FACT_ECG", "FACT_ECG_WAVEFORM", "FACT_ECG_WAVEFORM_RESTING_MEASUREMENT", "FACT_ECHO_TEST", "FACT_ECHO_RESULTS",
         "DIM_ECHO_TEST", "DIM_ECHO_TEST_RESULT_TYPE", "DIM_PATIENT", "FACT_RADIOLOGY", "FACT_IMAGINGSTUDY",
         "FACT_RADIOLOGY_DICOM_INVENTORY"]


def main():
    from google.cloud import bigquery

    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=None, help="BigQuery project holding the MCP datasets (default: the client's)")
    ap.add_argument("--out", default="exports/schema_inventory")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    client = bigquery.Client(project=a.project) if a.project else bigquery.Client()
    project = a.project or client.project
    rows = []
    for ds in client.list_datasets(project):
        for t in client.list_tables(ds.reference):
            tab = client.get_table(t.reference)
            cols = [f"{f.name} {f.field_type}" for f in tab.schema]
            rows.append(dict(project=project, dataset=ds.dataset_id, table=t.table_id, n_rows=tab.num_rows,
                             omop=t.table_id.lower() in OMOP,
                             modality_candidate=bool(MODALITY.search(t.table_id)),
                             n_columns=len(cols), columns=", ".join(cols)))
    inv = pd.DataFrame(rows)
    inv.to_csv(f"{a.out}/tables.csv", index=False)
    print(f"{len(inv)} tables in {inv.dataset.nunique() if len(inv) else 0} datasets of project {project}")
    watch = inv[inv.table.str.upper().isin(WATCH)]
    for _, r in watch.iterrows():
        print(f"\n{r.dataset}.{r.table} ({r.n_rows} rows)\n  {r.columns}")
    missing = sorted(set(WATCH) - set(inv.table.str.upper()))
    if missing:
        print(f"\nnot found: {', '.join(missing)}")
    person = inv[inv.table.str.lower() == "person"]
    if len(person):
        cdm = f"`{project}`.{person.dataset.iloc[0]}"
        print(f"\nOMOP CDM dataset detected: {cdm}  (config/study.yaml schemas.cdm / vocab)")

        def q(sql):
            return client.query(sql).result().to_dataframe()
        q(f"""SELECT cs.care_site_id, cs.care_site_name, l.state, COUNT(DISTINCT v.person_id) AS persons
              FROM {cdm}.visit_occurrence v JOIN {cdm}.care_site cs ON cs.care_site_id = v.care_site_id
              LEFT JOIN {cdm}.location l ON l.location_id = cs.location_id
              GROUP BY 1, 2, 3 ORDER BY persons DESC""").to_csv(f"{a.out}/care_sites.csv", index=False)
        q(f"SELECT note_title, COUNT(*) AS n FROM {cdm}.note GROUP BY note_title ORDER BY n DESC LIMIT 500") \
            .to_csv(f"{a.out}/note_titles.csv", index=False)
        q(f"SELECT vocabulary_id, COUNT(*) AS n FROM {cdm}.concept GROUP BY vocabulary_id ORDER BY n DESC") \
            .to_csv(f"{a.out}/vocabularies.csv", index=False)
        q(f"SELECT COUNTIF(year_of_birth = 1800) AS birth_year_redacted, COUNT(*) AS persons FROM {cdm}.person") \
            .to_csv(f"{a.out}/birth_year_redacted.csv", index=False)
        print(f"care sites, note titles, vocabularies and redacted birth years written to {a.out}/")


if __name__ == "__main__":
    main()
