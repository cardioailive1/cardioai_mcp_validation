"""BigQuery engine: MCP provisions the data in BigQuery (2026-10-28). The Spark SQL in this repo is translated by
sqlglot plus engine._bq_fix; these tests check the translation rules, the session flow with a stand-in client,
and that every cohort gives identical results when its SQL makes a round trip through BigQuery SQL."""
import sys
import types
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cardioai_mcp.engine import BigQueryEngine, DuckDBEngine, _one_group, spark_to_bigquery  # noqa: E402


def test_translation_rules():
    # Spark REGEXP_EXTRACT(s, p, g) -> one capturing group (GoogleSQL refuses two or more)
    assert _one_group("(lvef|ef)[^0-9]{0,25}([1-8][0-9]([.][0-9])?) ?%", 2) == "(?:lvef|ef)[^0-9]{0,25}([1-8][0-9](?:[.][0-9])?) ?%"
    assert _one_group(r"(?i)(a\(b)[(x](c)", 2) == r"(?i)(?:a\(b)[(x](c)"           # escapes, classes, (?i) untouched
    assert _one_group("(a)(b)", 0) == "(?:a)(?:b)"
    sql = spark_to_bigquery("SELECT REGEXP_EXTRACT(t, '(a)(b)', 2), REGEXP_EXTRACT(t, '(a)(b)'), YEAR(d), "
                            "GREATEST(x, y, NULL), LEAST(x, y), DATEDIFF(a, b), t RLIKE '(?i)\\\\btee\\\\b'")
    assert "REGEXP_EXTRACT(t, '(?:a)(b)')" in sql and "REGEXP_EXTRACT(t, '(a)(?:b)')" in sql
    assert "EXTRACT(YEAR FROM" in sql and "YEAR(" not in sql
    # Spark GREATEST / LEAST skip NULLs; GoogleSQL's return NULL -> MAX / MIN over UNNEST
    assert "(SELECT MAX(v) FROM UNNEST([x, y, NULL]) AS v)" in sql and "(SELECT MIN(v) FROM UNNEST([x, y]) AS v)" in sql
    assert "DATE_DIFF(" in sql and "REGEXP_CONTAINS(t, '(?i)\\\\btee\\\\b')" in sql
    assert "`ml-mps`.mcp.FACT_ECG" in spark_to_bigquery("SELECT * FROM `ml-mps`.mcp.FACT_ECG")
    with pytest.raises(ValueError, match="literal pattern"):
        spark_to_bigquery("SELECT REGEXP_EXTRACT(t, p, 1)")


class _Job:
    def __init__(self, sql, rows=None):
        self.sql = sql
        self.session_info = types.SimpleNamespace(session_id="sess-1")
        self._rows = rows

    def result(self):
        return self

    def to_dataframe(self):
        return self._rows if self._rows is not None else pd.DataFrame({"n": [1]})


class _Client:
    def __init__(self):
        self.calls, self.loads, self.tables = [], [], {"proj.work.locked"}

    def query(self, sql, job_config=None, location=None):
        self.calls.append((sql, job_config))
        return _Job(sql)

    def load_table_from_dataframe(self, df, name, job_config=None, location=None):
        self.loads.append((name, len(df), job_config.write_disposition))
        return _Job("load")

    def get_table(self, name):
        if name not in self.tables:
            raise KeyError(name)
        return name


@pytest.fixture
def fake_bigquery(monkeypatch):
    class _Cfg:
        def __init__(self, **kw):
            self.__dict__.update(kw)
    mod = types.ModuleType("google.cloud.bigquery")
    mod.QueryJobConfig = _Cfg
    mod.LoadJobConfig = _Cfg
    mod.ConnectionProperty = lambda k, v: (k, v)
    google = sys.modules.get("google") or types.ModuleType("google")
    cloud = types.ModuleType("google.cloud")
    cloud.bigquery = mod
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.cloud", cloud)
    monkeypatch.setitem(sys.modules, "google.cloud.bigquery", mod)
    return mod


def test_session_temp_tables_and_work_tables(fake_bigquery):
    client = _Client()
    eng = BigQueryEngine(client, {"schemas": {"cdm": "`proj`.omop", "work": "`proj`.work"}, "seed": 7})
    eng.create_view("echo_candidates", "SELECT person_id FROM {{cdm}}.person WHERE YEAR(d) > 2020")
    first, second = client.calls[0], client.calls[1]
    assert first[1].create_session is True                                  # one session for the whole run
    assert second[0].startswith("CREATE OR REPLACE TEMP TABLE echo_candidates AS SELECT person_id FROM `proj`.omop.person")
    assert second[1].connection_properties == [("session_id", "sess-1")]
    assert eng.table_exists("echo_candidates") and eng.table_exists("`proj`.work.locked") and not eng.table_exists("nope")
    assert eng.query("SELECT COUNT(*) AS n FROM echo_candidates").n[0] == 1
    assert eng.write_table(pd.DataFrame({"a": [1, 2]}), "echo_lvef") == "`proj`.work.echo_lvef"
    assert client.loads == [("proj.work.echo_lvef", 2, "WRITE_TRUNCATE")]      # backticks stripped for the API
    eng.save_view("echo_candidates", "echo_analytic_locked")
    assert client.calls[-1][0] == "CREATE OR REPLACE TABLE `proj`.work.echo_analytic_locked AS SELECT * FROM echo_candidates"
    assert "FARM_FINGERPRINT(CONCAT(" in spark_to_bigquery(eng.render("SELECT {{hash:person_id}}"))


def test_cohorts_identical_after_bigquery_round_trip():
    """Each cohort's SQL, translated Spark -> BigQuery -> DuckDB, gives the same tables as Spark -> DuckDB."""
    import sqlglot
    from cardioai_mcp.config import load_study
    from cardioai_mcp.pipeline import build_all
    from synthetic_omop import build

    class RoundTrip(DuckDBEngine):
        def _transpile(self, sql):
            return ";\n".join(sqlglot.transpile(spark_to_bigquery(sql), read="bigquery", write="duckdb"))

    def run(cls):
        st = load_study()
        st["schemas"].update(cdm="main", vocab="main", modality="main", work="main")
        return build_all(cls(build(1500), st), ["echo", "risk", "mri"], st, None)

    a, b = run(DuckDBEngine), run(RoundTrip)
    for k in a:
        x, y = a[k]["analytic"], b[k]["analytic"]
        assert a[k]["attrition"].equals(b[k]["attrition"]), k
        key = ["person_id"]
        x, y = x.sort_values(key).reset_index(drop=True), y.sort_values(key).reset_index(drop=True)
        num = x.select_dtypes("number").columns
        pd.testing.assert_frame_equal(x[num], y[num], check_dtype=False, check_exact=False, rtol=1e-9, obj=k)


def test_training_and_mcp_queries_translate():
    """Every training / MCP-table query renders and translates to BigQuery SQL (names as in the workspace)."""
    import ecg_training.dataset as ed
    import echo_wm_training.dataset as wd
    import mri_training.dataset as md
    from annotation.label_audit import troponin_concepts_sql, troponin_rows_sql
    from cardioai_mcp.config import load_study
    from ecg_training import mcp_waveforms as mw
    from ecg_training.extract_waveforms import waveform_batch_sql
    from echo_wm_training import mcp_echo as me
    from mri_training import mcp_imaging as mi
    st = load_study()
    st["schemas"].update(cdm="`p-1`.OMOP_v6_1", vocab="`p-1`.OMOP_v6_1", modality="`p-1`.mcp", work="`p-1`.cardioai_work")
    eng, mod = BigQueryEngine(None, st), st["modality"]
    sqls = [ed.evidence_sql(eng, st, ed.load_config(), mod), md.evidence_sql(eng, md.load_config(), mod),
            wd.evidence_sql(eng, mod), mw.ecg_index_sql(mod), mw.waveform_rows_sql(mod, ["C1"]), waveform_batch_sql(mod, ["1"]),
            me.echo_studies_sql(mod), me.lvef_rows_sql(mod), mi.cmr_studies_sql(mod), mi.series_sql(mod, ["1.2"]),
            troponin_rows_sql(mod, ["1"]), troponin_concepts_sql()]
    for s in sqls:
        import re
        out = re.sub(r"/\*.*?\*/", "", eng.transpile(s), flags=re.S)        # comments may mention {{macros}}
        assert out and "{{" not in out and "YEAR(" not in out and "GREATEST(" not in out
