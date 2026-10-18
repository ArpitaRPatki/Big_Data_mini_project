"""
SparkSession Factory Module
---------------------------
Viva Examination Notes:
1. Local Master (local[*]):
   - Runs Spark locally without a cluster manager (like YARN or Kubernetes).
   - The '*' wild card instructs Spark to allocate one executor worker thread per
     logical CPU core (12 threads on this Intel Core i5-1235U), maximizing parallelism.

2. Driver Memory Allocation (3g):
   - Total physical RAM detected is ~8.0 GB.
   - As per best practices, at least 4 GB is reserved for the operating system and background tasks.
   - Allocating 3 GB to the Spark driver prevents JVM OutOfMemory (OOM) errors while preventing OS thrashing.

3. Shuffle Partitions (spark.sql.shuffle.partitions = 8):
   - The Spark default is 200 partitions, which is tailored for large server clusters.
   - On a local laptop, 200 partitions causes severe scheduling overhead and thousands of tiny files.
   - Setting shuffle partitions to 8 (or core count) keeps wide transformations (groupBy, join) fast and lightweight.

4. Spark Web UI (spark.ui.enabled = true, port 4040):
   - Serves an interactive dashboard at http://localhost:4040.
   - Enables real-time inspection of active jobs, stages, execution DAGs (Directed Acyclic Graphs), and task timings.
"""

import os
import sys
from pathlib import Path
from pyspark.sql import SparkSession

# Ensure logs directory exists for any file appenders
from config import LOGS_DIR


def get_spark(app_name: str = "EcommerceAnalytics") -> SparkSession:
    """
    Initializes and returns a production-ready, locally optimized SparkSession.

    Parameters:
    -----------
    app_name : str
        Human-readable name for the Spark application (displayed in Spark Web UI).

    Returns:
    --------
    SparkSession
        Configured active Spark session.
    """
    # Fix for Windows native I/O if HADOOP_HOME is not globally registered
    # Checks local winutils directory if present
    project_root = Path(__file__).resolve().parent
    local_hadoop = project_root / "hadoop"
    if local_hadoop.exists() and "HADOOP_HOME" not in os.environ:
        os.environ["HADOOP_HOME"] = str(local_hadoop)
        os.environ["PATH"] = f"{local_hadoop / 'bin'};{os.environ.get('PATH', '')}"

    builder = (
        SparkSession.builder
        # 1. Master: use all logical CPU cores available on this machine
        .master("local[*]")
        # 2. Application Name: visible in UI and logs
        .appName(app_name)
        # 3. Driver Memory: 3 GB (leaving >4 GB for Windows OS and other apps)
        .config("spark.driver.memory", "3g")
        # 4. Shuffle Partitions: 8 partitions to eliminate 200-partition overhead on a laptop
        .config("spark.sql.shuffle.partitions", "8")
        # 5. Default Parallelism: aligns RDD operations with CPU cores
        .config("spark.default.parallelism", "8")
        # 6. Spark UI: Enable Web UI on port 4040 (http://localhost:4040)
        .config("spark.ui.enabled", "true")
        .config("spark.ui.port", "4040")
        # 7. Adaptive Query Execution (AQE): dynamically optimizes Spark SQL execution plans
        .config("spark.sql.adaptive.enabled", "true")
        # 8. Timezone consistency: UTC ensures consistent timestamp parsing for clickstream logs
        .config("spark.sql.session.timeZone", "UTC")
    )

    spark = builder.getOrCreate()

    # Set log level to WARN so console output is clean, readable, and free of verbose INFO spam
    spark.sparkContext.setLogLevel("WARN")

    return spark


if __name__ == "__main__":
    # Quick standalone test of session creation
    print("Testing SparkSession initialization...")
    spark = get_spark("TestSession")
    print(f"Spark version: {spark.version}")
    print(f"Master: {spark.sparkContext.master}")
    print(f"Driver Memory: {spark.conf.get('spark.driver.memory')}")
    print(f"Shuffle Partitions: {spark.conf.get('spark.sql.shuffle.partitions')}")
    print(f"Spark UI URL: {spark.sparkContext.uiWebUrl}")
    spark.stop()
    print("SparkSession stopped cleanly.")
