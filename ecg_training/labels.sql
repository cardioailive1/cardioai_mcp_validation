-- One row per ECG with the OMOP evidence the labels are built from (labels.py turns it into heads).
-- Macros: {{cdm}}, {{cs:set}}, {{ecg_table}}, {{c_<column>}} (from config/modality_tables.yaml), {{win_lo}}, {{win_hi}}, {{clear_days}}.
WITH e AS (
  SELECT CAST(x.{{c_person_id}} AS BIGINT)                 AS person_id,
         CAST(x.{{c_ecg_id}} AS STRING)                    AS ecg_id,
         CAST(x.{{c_acquisition_datetime}} AS DATE)        AS ecg_date,
         {{waveform_select}}                               AS waveform_uri,   -- NULL for Base64 storage: filled by extract_waveforms
         x.{{c_read_statement}}                            AS read_text,
         CAST(x.{{c_is_confirmed}} AS INT)                 AS confirmed
  FROM {{ecg_table}} x
  WHERE x.{{c_waveform_uri}} IS NOT NULL
    AND CAST(x.{{c_acquisition_datetime}} AS DATE) BETWEEN CAST('{{index_start}}' AS DATE) AND CAST('{{data_cutoff}}' AS DATE)
)
SELECT e.*,
       COALESCE(p.care_site_id, -1)                        AS site,
       YEAR(e.ecg_date) - p.year_of_birth                  AS age,
       p.gender_concept_id,
       CASE WHEN EXISTS (SELECT 1 FROM {{cdm}}.condition_occurrence co WHERE co.person_id = e.person_id
              AND DATEDIFF(co.condition_start_date, e.ecg_date) BETWEEN {{win_lo}} AND {{win_hi}}
              AND (co.condition_concept_id IN {{cs:mi_acute}} OR co.condition_source_concept_id IN {{cs:mi_acute}}))
            THEN 1 ELSE 0 END                              AS mi_dx,
       CASE WHEN EXISTS (SELECT 1 FROM {{cdm}}.condition_occurrence co WHERE co.person_id = e.person_id
              AND DATEDIFF(co.condition_start_date, e.ecg_date) BETWEEN {{win_lo}} AND {{win_hi}}
              AND (co.condition_concept_id IN {{cs:stemi}} OR co.condition_source_concept_id IN {{cs:stemi}}))
            THEN 1 ELSE 0 END                              AS stemi_dx,
       CASE WHEN EXISTS (SELECT 1 FROM {{cdm}}.condition_occurrence co WHERE co.person_id = e.person_id
              AND DATEDIFF(co.condition_start_date, e.ecg_date) BETWEEN {{win_lo}} AND {{win_hi}}
              AND (co.condition_concept_id IN {{cs:nstemi}} OR co.condition_source_concept_id IN {{cs:nstemi}}))
            THEN 1 ELSE 0 END                              AS nstemi_dx,
       CASE WHEN EXISTS (SELECT 1 FROM {{cdm}}.condition_occurrence co WHERE co.person_id = e.person_id
              AND ABS(DATEDIFF(co.condition_start_date, e.ecg_date)) <= {{clear_days}}
              AND (co.condition_concept_id IN {{cs:mi_any}} OR co.condition_source_concept_id IN {{cs:mi_any}}))
            THEN 1 ELSE 0 END                              AS mi_any_near,
       CASE WHEN EXISTS (SELECT 1 FROM {{cdm}}.measurement m WHERE m.person_id = e.person_id
              AND DATEDIFF(m.measurement_date, e.ecg_date) BETWEEN {{win_lo}} AND {{win_hi}}
              AND (m.measurement_concept_id IN {{cs:troponin}} OR m.measurement_source_concept_id IN {{cs:troponin}})
              AND m.value_as_number > COALESCE(m.range_high,
                    CASE WHEN LOWER(TRIM(m.unit_source_value)) IN ('ng/l', 'pg/ml') THEN
                              CASE p.gender_concept_id WHEN 8507 THEN {{trop_ng_per_l_male}}
                                                       WHEN 8532 THEN {{trop_ng_per_l_female}}
                                                       ELSE {{trop_ng_per_l}} END
                         WHEN LOWER(TRIM(m.unit_source_value)) IN ('ng/ml', 'ug/l') THEN {{trop_ng_per_ml}} END))
            THEN 1 ELSE 0 END                              AS trop_pos
FROM e JOIN {{cdm}}.person p ON p.person_id = e.person_id
