"""Inventory the MCP workspace: OMOP tables, modality tables, row counts, care sites, note titles.

Run once at onboarding, then fill config/modality_tables.yaml and config/study.yaml (site_map):
    python scripts/discover_schema.py --out exports/schema_inventory
"""
from __future__ import annotations

import argparse
import os
import re

import pandas as pd

OMOP = ["person", "observation_period", "visit_occurrence", "condition_occurrence", "procedure_occurrence", "drug_exposure",
        "measurement", "observation", "device_exposure", "death", "note", "note_nlp", "care_site", "location",
        "concept", "concept_relationship", "concept_ancestor"]
MODALITY = re.compile(r"(ecg|ekg|waveform|wave|echo|dicom|image|imaging|study|series|cine)", re.I)


def main():
    from pyspark.sql import SparkSession

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="exports/schema_inventory")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    spark = SparkSession.builder.getOrCreate()
    rows = []
    for db in [r.namespace if hasattr(r, "namespace") else r[0] for r in spark.sql("SHOW DATABASES").collect()]:
        for t in spark.sql(f"SHOW TABLES IN {db}").collect():
            name = t.tableName
            cols = [f.name for f in spark.table(f"{db}.{name}").schema.fields]
            rows.append(dict(database=db, table=name, omop=name.lower() in OMOP, modality_candidate=bool(MODALITY.search(name)) or
                             any(MODALITY.search(c) for c in cols), n_columns=len(cols), columns=", ".join(cols)))
    inv = pd.DataFrame(rows)
    inv.to_csv(f"{a.out}/tables.csv", index=False)
    print(inv[inv.modality_candidate][["database", "table", "columns"]].to_string(index=False))
    cdm = inv[inv.table.str.lower() == "person"].database.iloc[0] if (inv.table.str.lower() == "person").any() else None
    if cdm:
        print(f"\nOMOP CDM schema detected: {cdm}")
        spark.sql(f"""SELECT cs.care_site_id, cs.care_site_name, l.state, COUNT(DISTINCT v.person_id) AS persons
                      FROM {cdm}.visit_occurrence v JOIN {cdm}.care_site cs ON cs.care_site_id = v.care_site_id
                      LEFT JOIN {cdm}.location l ON l.location_id = cs.location_id
                      GROUP BY 1,2,3 ORDER BY persons DESC""").toPandas().to_csv(f"{a.out}/care_sites.csv", index=False)
        spark.sql(f"""SELECT note_title, COUNT(*) AS n FROM {cdm}.note GROUP BY note_title ORDER BY n DESC LIMIT 500""") \
            .toPandas().to_csv(f"{a.out}/note_titles.csv", index=False)
        spark.sql(f"""SELECT vocabulary_id, COUNT(*) AS n FROM {cdm}.concept GROUP BY vocabulary_id ORDER BY n DESC""") \
            .toPandas().to_csv(f"{a.out}/vocabularies.csv", index=False)
        print(f"care sites, note titles and vocabularies written to {a.out}/")


if __name__ == "__main__":
    main()
