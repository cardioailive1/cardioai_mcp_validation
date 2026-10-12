"""Execution engines.

All SQL in this package is written in **Spark SQL** — the dialect of the MCP
workspace (SparkSQL API from JupyterLab).  ``DuckDBEngine`` transpiles the same
SQL with sqlglot so the full pipeline can be unit-tested offline on synthetic
OMOP data before it ever touches MCP.
"""
from __future__ import annotations

import re
from typing import Any

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
