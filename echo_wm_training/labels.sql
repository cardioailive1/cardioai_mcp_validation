-- One row per echo study with its report text and DICOM folder (dataset.py makes the wall-motion labels).
-- Macros: {{cdm}}, {{echo_wm_table}}, {{c_<column>}} (config/modality_tables.yaml: echo_wm.study_columns).
WITH s AS (
  SELECT CAST(x.{{c_person_id}} AS BIGINT)            AS person_id,
         CAST(x.{{c_study_id}} AS STRING)             AS study_id,
         CAST(x.{{c_study_datetime}} AS DATE)         AS study_date,
         x.{{c_dicom_uri}}                            AS dicom_uri,
         x.{{c_report_text}}                          AS report_text
  FROM {{echo_wm_table}} x
  WHERE x.{{c_dicom_uri}} IS NOT NULL
    AND CAST(x.{{c_study_datetime}} AS DATE) BETWEEN CAST('{{index_start}}' AS DATE) AND CAST('{{data_cutoff}}' AS DATE)
)
SELECT s.*,
       COALESCE(p.care_site_id, -1)                   AS site,
       YEAR(s.study_date) - p.year_of_birth           AS age,
       p.gender_concept_id
FROM s JOIN {{cdm}}.person p ON p.person_id = s.person_id
