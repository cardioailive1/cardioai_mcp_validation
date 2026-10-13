"""Run the real Spark SQL (no transpilation) on a local Spark session over synthetic OMOP.

    PYTHONPATH=<pyspark> python tests/spark_local_check.py
Synthetic OMOP tables are generated in DuckDB and registered as Spark global temp views,
so {{cdm}} = global_temp.  Every cohort is built through SparkEngine exactly as on MCP.
"""
import os
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from pyspark.sql import SparkSession  # noqa: E402

from cardioai_mcp import COHORTS  # noqa: E402
from cardioai_mcp.config import load_study  # noqa: E402
from cardioai_mcp.engine import SparkEngine  # noqa: E402
from cardioai_mcp.pipeline import build_all  # noqa: E402
from tests.synthetic_omop import build  # noqa: E402

t0 = time.time()
spark = (SparkSession.builder.master("local[2]").appName("cardioai-check")
         .config("spark.sql.shuffle.partitions", "8").config("spark.sql.warehouse.dir", tempfile.mkdtemp(prefix="spark_wh_")).config("spark.driver.memory", os.environ.get("DRIVER_MEM", "4g"))
         .config("spark.sql.execution.arrow.pyspark.enabled", "true").getOrCreate())
spark.sparkContext.setLogLevel("ERROR")
con = build(int(os.environ.get("N_PERSONS", "3000")))

pq = tempfile.mkdtemp(prefix="omop_pq_")
for (t,) in con.execute("SELECT table_name FROM information_schema.tables WHERE table_schema='main'").fetchall():
    con.execute(f"COPY {t} TO '{pq}/{t}.parquet' (FORMAT PARQUET)")  # typed hand-off DuckDB -> Spark
    spark.read.parquet(f"{pq}/{t}.parquet").createOrReplaceGlobalTempView(t)
print(f"loaded synthetic OMOP into Spark in {time.time() - t0:.0f}s")
study = load_study()
study["schemas"].update(cdm="global_temp", vocab="global_temp", modality="global_temp", work="default")
eng = SparkEngine(spark, study, materialize=True)
res = build_all(eng, os.environ.get("COHORTS", " ".join(COHORTS)).split(), study, os.path.join(ROOT, "exports", "spark_local_check"))
for k, r in res.items():
    a = r["analytic"]
    print(f"{k}: {len(a)} rows x {a.shape[1]} cols; last attrition step: {r['attrition'].iloc[-1]['step'][:90]}")
print(f"Spark SQL check OK in {time.time() - t0:.0f}s")
