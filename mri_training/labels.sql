-- One row per cardiac MRI study with its report text and clinical context (dataset.py makes the labels).
-- Macros: {{cdm}}, {{cs:set}}, {{mri_table}}, {{c_<column>}} (config/modality_tables.yaml: mri.columns).
WITH s AS (
  SELECT CAST(x.{{c_person_id}} AS BIGINT)            AS person_id,
         CAST(x.{{c_study_id}} AS STRING)             AS study_id,
         CAST(x.{{c_study_datetime}} AS DATE)         AS study_date,
         x.{{c_dicom_uri}}                            AS dicom_uri,
         x.{{c_report_text}}                          AS report_text
  FROM {{mri_table}} x
  WHERE x.{{c_dicom_uri}} IS NOT NULL
    AND CAST(x.{{c_study_datetime}} AS DATE) BETWEEN CAST('{{index_start}}' AS DATE) AND CAST('{{data_cutoff}}' AS DATE)
)
SELECT s.*,
       COALESCE(p.care_site_id, -1)                   AS site,
       YEAR(s.study_date) - NULLIF(p.year_of_birth, 1800)           AS age,
       p.gender_concept_id,
       CASE WHEN EXISTS (SELECT 1 FROM {{cdm}}.condition_occurrence co WHERE co.person_id = s.person_id
              AND DATEDIFF(s.study_date, co.condition_start_date) BETWEEN 0 AND 30
              AND (co.condition_concept_id IN {{cs:stemi}} OR co.condition_source_concept_id IN {{cs:stemi}}))
            THEN 1 ELSE 0 END                         AS recent_stemi,
       CASE WHEN EXISTS (SELECT 1 FROM {{cdm}}.condition_occurrence co WHERE co.person_id = s.person_id
              AND DATEDIFF(s.study_date, co.condition_start_date) BETWEEN 0 AND 30
              AND (co.condition_concept_id IN {{cs:mi_acute}} OR co.condition_source_concept_id IN {{cs:mi_acute}}))
            THEN 1 ELSE 0 END                         AS recent_acute_mi,
       CASE WHEN EXISTS (SELECT 1 FROM {{cdm}}.procedure_occurrence po WHERE po.person_id = s.person_id
              AND DATEDIFF(po.procedure_date, s.study_date) BETWEEN -1 AND 1
              AND (po.procedure_concept_id IN {{cs:cardiac_mri_contrast}} OR po.procedure_source_concept_id IN {{cs:cardiac_mri_contrast}}))
            THEN 1 ELSE 0 END                         AS contrast_billed
FROM s JOIN {{cdm}}.person p ON p.person_id = s.person_id
