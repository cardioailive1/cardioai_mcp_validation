"""MCP radiology tables (Radiology slides, MCP Resource Hub): find CMR studies with final reports, request only
the short-axis LGE + cine series that are de-identifiable, and read the delivered DICOM folders."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
pytest.importorskip("pydicom")
duckdb = pytest.importorskip("duckdb")

from cardioai_mcp.config import load_study  # noqa: E402
from cardioai_mcp.engine import DuckDBEngine  # noqa: E402
from mri_training import mcp_imaging as mi  # noqa: E402

SERIES = [  # (description, instances, deid status)
    ("localizer", 3, "Ready"), ("cine_4ch SSFP", 25, "Ready"), ("SA cine SSFP", 250, "Ready"),
    ("SA LGE PSIR", 10, "Available"), ("4CH LGE PSIR", 1, "Ready"), ("T1 MAP MOLLI", 8, "Ready"),
]


def _db():
    st = load_study()
    st["schemas"].update(cdm="main", vocab="main", modality="main", work="main")
    con = duckdb.connect()
    # OMOP person: person_source_value = PATIENT_CLINIC_NUMBER (MCP 2026-10-28)
    con.execute("CREATE TABLE person AS SELECT * FROM (VALUES (11, 'C1'), (12, 'C2'), (13, 'C3'), (14, 'C4'), (15, 'C5')) v(person_id, person_source_value)")
    con.execute("""CREATE TABLE FACT_IMAGINGSTUDY (PATIENT_CLINIC_NUMBER VARCHAR, STUDY_INSTANCE_UID VARCHAR, STMO_NAME_IN_ADL VARCHAR,
                   MODALITY_1 VARCHAR, MODALITY_2 VARCHAR, PROCEDURE_CODE_TEXT VARCHAR, PROCEDURE_CODE_CODING_CODE VARCHAR,
                   STUDY_DESCRIPTION VARCHAR, NUMBER_OF_SERIES INT, NUMBER_OF_INSTANCES INT)""")
    con.execute("""CREATE TABLE FACT_RADIOLOGY_DICOM_INVENTORY (PATIENT_CLINIC_NUMBER VARCHAR, DICOM_ACCESSION_NUMBER VARCHAR,
                   STMO_NAME_IN_ADL VARCHAR, MODALITY VARCHAR, STUDY_INSTANCE_UID VARCHAR, SERIES_INSTANCE_UID VARCHAR,
                   SOP_INSTANCE_UID VARCHAR, STUDY_DESCRIPTION VARCHAR, SERIES_DESCRIPTION VARCHAR, IMAGE_DEID_STATUS VARCHAR)""")
    con.execute("""CREATE TABLE FACT_RADIOLOGY (ACCESSION_NBR VARCHAR, PATIENT_DK BIGINT, RADIOLOGY_DTM TIMESTAMP, RADIOLOGY_REPORT VARCHAR,
                   RADIOLOGY_IMPRESSION VARCHAR, SERVICE_MODALITY_CODE VARCHAR, SERVICE_RESULT_STATUS VARCHAR)""")
    studies = [  # uid, clinic, cpt, description, status, report, series deid override
        ("1.1", "C1", "75561", "MR HEART WITH CONTRAST", "Final", "Subendocardial LGE consistent with infarction.", None),
        ("1.2", "C2", "70553", "MR CARDIAC VIABILITY", "Final", "No late gadolinium enhancement.", None),
        ("1.3", "C3", "75561", "MR HEART WITH CONTRAST", "Preliminary", "Draft.", None),              # not final
        ("1.4", "C4", "70553", "MR BRAIN WITH CONTRAST", "Final", "Normal brain.", None),             # not cardiac
        ("1.5", "C5", "75561", "MR HEART WITH CONTRAST", "Final", "LGE report.", "Not Available"),   # not de-identifiable
    ]
    for k, (uid, cn, cpt, desc, status, rep, deid) in enumerate(studies):
        acc = f"A{k}"
        con.execute("INSERT INTO FACT_IMAGINGSTUDY VALUES (?, ?, 'STMO1', 'MR', NULL, ?, ?, ?, 6, 297)", [cn, uid, desc, cpt, desc])
        con.execute("INSERT INTO FACT_RADIOLOGY VALUES (?, ?, TIMESTAMP '2022-03-01 10:00:00', ?, 'Impression.', 'MR', ?)",
                    [acc, 1000 + k, rep, status])
        for j, (sd, n, st_) in enumerate(SERIES):
            for i in range(n):
                con.execute("INSERT INTO FACT_RADIOLOGY_DICOM_INVENTORY VALUES (?, ?, 'STMO1', 'MR', ?, ?, ?, ?, ?, ?)",
                            [cn, acc, uid, f"{uid}.{j}", f"{uid}.{j}.{i}", desc, sd, deid or st_])
    return con, st, DuckDBEngine(con, st)


def test_find_request_and_build_study_table(tmp_path):
    import synthetic_cmr as S
    from mri_training.io import read_study_path
    con, st, eng = _db()
    studies = mi.find_cmr_studies(eng, st["modality"])
    assert sorted(studies.study_uid) == ["1.1", "1.2", "1.5"]                     # cardiac + final report only
    assert "infarction" in studies.set_index("study_uid").loc["1.1", "report_text"]
    assert dict(zip(studies.study_uid, studies.person_id.astype(int))) == {"1.1": 11, "1.2": 12, "1.5": 15}
    series = mi.list_series(eng, st["modality"], studies)
    req, rep = mi.build_request(series)
    assert set(req.study_uid) == {"1.1", "1.2"} and rep["studies_without_lge_or_not_deidentifiable"] == 1
    assert sorted(req.series_description.unique()) == ["SA LGE PSIR", "SA cine SSFP"]   # no localiser, 4ch, mapping
    assert rep["instances_requested"] == 2 * (250 + 10) < rep["instances_total"]
    deid = mi.deid_request(studies, req)                                          # the file MCP asked for
    assert list(deid.columns) == ["STUDY_INSTANCE_UID", "PATIENT_DK", "MODALITY", "STMO_NAME_IN_ADL"]
    assert sorted(deid.STUDY_INSTANCE_UID) == ["1.1", "1.2"] and set(deid.MODALITY) == {"MR"}
    assert sorted(deid.PATIENT_DK) == [1000, 1001] and set(deid.STMO_NAME_IN_ADL) == {"STMO1"}
    # delivery: one folder per study (here written as synthetic DICOM)
    lge, cine, _ = S.study(seed=3, n_slices=6, phases=8)
    import io
    import zipfile
    z = zipfile.ZipFile(io.BytesIO(S.dicom_zip(lge, cine)))
    d = tmp_path / "delivered" / "batch1" / "1.1"
    d.mkdir(parents=True)
    for n in z.namelist():
        (d / Path(n).name).write_bytes(z.read(n))
    cmr = mi.build_study_table(studies, str(tmp_path / "delivered"))
    assert cmr.study_uid.tolist() == ["1.1"] and cmr.dicom_path[0].endswith("1.1")
    l2, c2 = read_study_path(cmr.dicom_path[0])
    assert l2.images.shape[0] == 6 and c2.images.shape[:2] == (8, 6)


def test_series_rules():
    assert mi.classify_series("SA LGE PSIR", 10) == "lge"
    assert mi.classify_series("PSIR 3CH", 1) == "other"
    assert mi.classify_series("sax cine FIESTA", 300) == "cine"
    assert mi.classify_series("cine 4ch", 25) == "other"
    assert mi.classify_series("T1 MAP MOLLI SA", 8) == "other"
    assert mi.classify_series("localizer", 3) == "other"
