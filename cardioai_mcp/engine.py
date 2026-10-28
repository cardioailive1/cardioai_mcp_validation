"""Execution engines.

All SQL in this package is written in **Spark SQL**. Mayo Clinic Platform (Adam Resnick / MCP, 2026-10-28)
provisions the data in **BigQuery** and the workspace queries it with the BigQuery client, so ``BigQueryEngine``
transpiles every statement to GoogleSQL with sqlglot (plus the fixes in ``_bq_fix``) and runs it in one BigQuery
session, where the pipeline's intermediate views are session temp tables. ``SparkEngine`` is kept for a Spark
workspace; ``DuckDBEngine`` transpiles the same SQL to DuckDB so the pipeline is tested offline on synthetic OMOP.

    eng = make_engine("bigquery", study)        # in the MCP workspace
    df = eng.query("SELECT COUNT(*) AS n FROM {{cdm}}.person")
    eng.write_table(df, "my_table")             # -> {work}.my_table
"""
from __future__ import annotations

import re
from typing import Any, Optional

import pandas as pd

_MACRO = re.compile(r"\{\{\s*([a-z_]+)(?::([^}]*))?\s*\}\}")


class Engine:
    dialect = "spark"

    def __init__(self, study: dict):
        self.study = study
        self.schemas = dict(study.get("schemas", {}))
        self.log: list[tuple[str, str]] = []

    # ---------------------------------------------------------------- macros
    def render(self, sql: str, extra: dict | None = None) -> str:
        """Expand {{...}} macros.

        {{cdm}} {{vocab}} {{work}} {{modality}}  schema names
        {{cs:name}}        sub-query of concept ids in concept set `name`
        {{hash:expr}}      deterministic 64-bit hash (seeded) for sampling
        {{index_start}} {{index_end}} {{data_cutoff}} {{min_age}} {{seed}} ...
        """
        values = {**self.schemas, **{k: v for k, v in self.study.items() if isinstance(v, (str, int, float))}}
        if extra:
            values.update(extra)

        def sub(m: re.Match) -> str:
            key, arg = m.group(1), m.group(2)
            if key == "cs":
                return f"(SELECT concept_id FROM cs_members WHERE set_name = '{arg.strip()}')"
            if key == "hash":
                return self.hash_expr(arg.strip())
            if key not in values:
                raise KeyError(f"unknown SQL macro {{{{{key}}}}}")
            return str(values[key])

        prev = None
        while prev != sql:  # allow nested macros (e.g. modality table names)
            prev, sql = sql, _MACRO.sub(sub, sql)
        return sql

    def hash_expr(self, expr: str) -> str:
        return f"xxhash64(CAST({expr} AS STRING), '{self.study.get('seed', 0)}')"

    # ---------------------------------------------------------------- API
    def create_view(self, name: str, sql: str) -> None:
        raise NotImplementedError

    def query(self, sql: str) -> pd.DataFrame:
        raise NotImplementedError

    def table_exists(self, name: str) -> bool:
        raise NotImplementedError

    def write_table(self, df: pd.DataFrame, name: str) -> str:
        """Save a DataFrame as {work}.name (replacing it); returns the full name."""
        raise NotImplementedError

    def read_table(self, name: str) -> pd.DataFrame:
        """A saved table ({work}.name, or a full name) or a pipeline view, as a DataFrame."""
        for full in ((name,) if "." in name else (self.work_name(name), name)):
            if self.table_exists(full):
                return self.query(f"SELECT * FROM {full}")
        raise KeyError(f"table {name} not found in the work schema or as a view")

    def save_view(self, view: str, name: Optional[str] = None) -> str:
        """Persist a pipeline view (e.g. echo_analytic) as {work}.name."""
        raise NotImplementedError

    def work_name(self, name: str) -> str:
        return f"{self.schemas['work']}.{name}"


class SparkEngine(Engine):
    """Runs inside the MCP JupyterLab workspace (``spark`` session provided)."""

    def __init__(self, spark: Any, study: dict, materialize: str | bool = "checkpoint"):
        """materialize: 'checkpoint' (default) — eager localCheckpoint(); truncates lineage so later
        queries don't re-plan every upstream step (big win for the many-step cohort pipeline);
        'cache' — lazy cache; 'table' — write to the work schema; False — plain views."""
        super().__init__(study)
        self.spark = spark
        self.materialize = "checkpoint" if materialize is True else materialize

    def create_view(self, name: str, sql: str) -> None:
        sql = self.render(sql)
        self.log.append((name, sql))
        df = self.spark.sql(sql)
        if self.materialize == "checkpoint":
            df = df.localCheckpoint(eager=True)
        elif self.materialize == "cache":
            df = df.cache()
        elif self.materialize == "table":
            full = f"{self.schemas['work']}.{name}"
            df.write.mode("overwrite").saveAsTable(full)
            df = self.spark.table(full)
        df.createOrReplaceTempView(name)

    def query(self, sql: str) -> pd.DataFrame:
        sql = self.render(sql)
        self.log.append(("<query>", sql))
        return self.spark.sql(sql).toPandas()

    def table_exists(self, name: str) -> bool:
        try:
            self.spark.table(name)
            return True
        except Exception:  # noqa: BLE001
            return False

    def write_table(self, df: pd.DataFrame, name: str) -> str:
        full = self.work_name(name)
        self.spark.createDataFrame(df).write.mode("overwrite").saveAsTable(full)
        return full

    def save_view(self, view: str, name: Optional[str] = None) -> str:
        full = self.work_name(name or view)
        self.spark.table(view).write.mode("overwrite").saveAsTable(full)
        return full


# ------------------------------------------------------------------------------------------------ BigQuery
_GROUP = __import__("re").compile(r"\\.|\[(?:\\.|[^\]])*\]|\((?!\?)")


def _one_group(pattern: str, keep: int) -> str:
    """Spark REGEXP_EXTRACT(s, p, g) returns group g; GoogleSQL REGEXP_EXTRACT(s, p) returns the ONLY capturing
    group (or the whole match when there is none) and refuses patterns with two or more. Every capturing group
    except number `keep` becomes non-capturing (keep = 0: all of them)."""
    out, n, pos = [], 0, 0
    for m in _GROUP.finditer(pattern):
        if m.group() != "(":
            continue                                   # an escape or a character class: left alone
        n += 1
        out.append(pattern[pos:m.start()])
        out.append("(" if n == keep else "(?:")
        pos = m.end()
    out.append(pattern[pos:])
    return "".join(out)


def _bq_fix(node):
    """Spark constructs sqlglot does not carry over to GoogleSQL (applied until nothing changes):
    YEAR/MONTH/DAY -> EXTRACT; REGEXP_EXTRACT group -> a single capturing group; GREATEST/LEAST skip NULLs in
    Spark but return NULL in GoogleSQL when any argument is NULL -> MAX/MIN over UNNEST (NULLs skipped)."""
    import sqlglot
    from sqlglot import exp
    if isinstance(node, (exp.Greatest, exp.Least)):
        args = [node.this] + list(node.expressions or [])
        if len(args) > 1:
            agg = "MAX" if isinstance(node, exp.Greatest) else "MIN"
            sub = sqlglot.parse_one(f"(SELECT {agg}(v) FROM UNNEST([0]) AS v)", read="bigquery")
            sub.find(exp.Array).set("expressions", [a.copy() for a in args])
            return sub
    if isinstance(node, (exp.Year, exp.Month, exp.Day)):
        part = {exp.Year: "YEAR", exp.Month: "MONTH", exp.Day: "DAY"}[type(node)]
        return exp.Extract(this=exp.var(part), expression=node.this)
    if isinstance(node, exp.RegexpExtract):
        g = node.args.get("group")
        keep = int(g.name) if g is not None and g.is_int else (1 if g is None else None)
        pat = node.expression
        if keep is None or not isinstance(pat, exp.Literal) or not pat.is_string:
            raise ValueError(f"REGEXP_EXTRACT needs a literal pattern and group for BigQuery: {node.sql('spark')}")
        return exp.RegexpExtract(this=node.this, expression=exp.Literal.string(_one_group(pat.name, keep)))
    return node


def spark_to_bigquery(sql: str) -> str:
    import sqlglot
    from sqlglot.errors import ErrorLevel
    out = []
    for tree in sqlglot.parse(sql, read="spark"):
        if tree is None:
            continue
        prev = None
        while prev != tree.sql("bigquery"):            # nested fixes (e.g. YEAR inside GREATEST) need a second pass
            prev = tree.sql("bigquery")
            tree = tree.transform(_bq_fix)
        out.append(tree.sql("bigquery", unsupported_level=ErrorLevel.RAISE))
    return ";\n".join(out)


class BigQueryEngine(Engine):
    """MCP workspace: Spark SQL -> GoogleSQL (sqlglot), run with the BigQuery client in one session.

    Pipeline views (cs_members, <cohort>_candidates, ...) are session TEMP tables, referenced by their bare names.
    Schema macros in config/study.yaml are BigQuery datasets: `project`.dataset (backticks when the project id
    has hyphens), e.g. cdm: "`<project>`.<omop dataset>", work: "`<project>`.cardioai_work"."""

    dialect = "bigquery"

    def __init__(self, client: Any, study: dict, location: Optional[str] = None):
        super().__init__(study)
        self.client = client
        self.location = location
        self.session_id: Optional[str] = None
        self.views: set = set()

    def hash_expr(self, expr: str) -> str:
        # passes through the Spark parser as an unknown function and comes out unchanged
        return f"FARM_FINGERPRINT(CONCAT(CAST({expr} AS STRING), '{self.study.get('seed', 0)}'))"

    def _config(self):
        from google.cloud import bigquery
        if self.session_id is None:
            job = self.client.query("SELECT 1", job_config=bigquery.QueryJobConfig(create_session=True),
                                    location=self.location)
            job.result()
            self.session_id = job.session_info.session_id
        return bigquery.QueryJobConfig(connection_properties=[bigquery.ConnectionProperty("session_id", self.session_id)])

    def _run(self, sql: str):
        return self.client.query(sql, job_config=self._config(), location=self.location).result()

    def transpile(self, sql: str) -> str:
        return spark_to_bigquery(self.render(sql))

    def create_view(self, name: str, sql: str) -> None:
        sql = self.transpile(sql)
        self.log.append((name, sql))
        self._run(f"CREATE OR REPLACE TEMP TABLE {name} AS {sql}")
        self.views.add(name)

    def query(self, sql: str) -> pd.DataFrame:
        sql = self.transpile(sql)
        self.log.append(("<query>", sql))
        return self._run(sql).to_dataframe()

    @staticmethod
    def _api_name(full: str) -> str:
        return full.replace("`", "")

    def table_exists(self, name: str) -> bool:
        if name in self.views:
            return True
        if "." not in name:
            return False
        try:
            self.client.get_table(self._api_name(self.render(name)))
            return True
        except Exception:  # noqa: BLE001
            return False

    def write_table(self, df: pd.DataFrame, name: str) -> str:
        from google.cloud import bigquery
        full = self.render(self.work_name(name))
        cfg = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
        self.client.load_table_from_dataframe(df, self._api_name(full), job_config=cfg, location=self.location).result()
        return full

    def save_view(self, view: str, name: Optional[str] = None) -> str:
        full = self.render(self.work_name(name or view))
        self._run(f"CREATE OR REPLACE TABLE {full} AS SELECT * FROM {view}")
        return full

    def close(self) -> None:
        if self.session_id:
            self._run("CALL BQ.ABORT_SESSION()")
            self.session_id = None


class DuckDBEngine(Engine):
    """Offline test engine: Spark SQL -> DuckDB via sqlglot."""

    dialect = "duckdb"

    def __init__(self, con: Any, study: dict):
        super().__init__(study)
        self.con = con
        try:  # works around a DuckDB 1.5 internal error on ROW_NUMBER()=1 filters (test engine only)
            self.con.execute("SET disabled_optimizers = 'top_n_window_elimination'")
        except Exception:  # noqa: BLE001
            pass

    def hash_expr(self, expr: str) -> str:
        # written in Spark syntax: transpiled afterwards; hash() is DuckDB-native
        return f"HASH(CAST({expr} AS STRING) || '{self.study.get('seed', 0)}')"

    def _transpile(self, sql: str) -> str:
        import sqlglot

        out = sqlglot.transpile(sql, read="spark", write="duckdb")
        return ";\n".join(out)

    def create_view(self, name: str, sql: str) -> None:
        sql = self._transpile(self.render(sql))
        self.log.append((name, sql))
        self.con.execute(f"CREATE OR REPLACE TEMP TABLE {name} AS {sql}")

    def query(self, sql: str) -> pd.DataFrame:
        sql = self._transpile(self.render(sql))
        self.log.append(("<query>", sql))
        return self.con.execute(sql).df()

    def table_exists(self, name: str) -> bool:
        try:
            self.con.execute(f"SELECT 1 FROM {name} LIMIT 0")
            return True
        except Exception:  # noqa: BLE001
            return False

    def write_table(self, df: pd.DataFrame, name: str) -> str:
        full = self.work_name(name)
        self.con.register("_df_in", df)
        self.con.execute(f"CREATE OR REPLACE TABLE {full} AS SELECT * FROM _df_in")
        self.con.unregister("_df_in")
        return full

    def save_view(self, view: str, name: Optional[str] = None) -> str:
        full = self.work_name(name or view)
        if full != view:
            self.con.execute(f"CREATE OR REPLACE TABLE {full} AS SELECT * FROM {view}")
        return full


def make_engine(kind: str, study: dict, **kw) -> Engine:
    """'bigquery' (MCP workspace: google.cloud.bigquery.Client()), 'spark' (a Spark workspace)."""
    if kind == "bigquery":
        from google.cloud import bigquery
        return BigQueryEngine(kw.pop("client", None) or bigquery.Client(project=study.get("bq_project")), study, **kw)
    if kind == "spark":
        from pyspark.sql import SparkSession
        return SparkEngine(SparkSession.builder.getOrCreate(), study, **kw)
    raise ValueError(f"unknown engine {kind!r}")
