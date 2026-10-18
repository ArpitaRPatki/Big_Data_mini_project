"""
SparkSession Factory Module
---------------------------
Viva Examination Notes:
1. Local Master (local[*]):
   - Runs Spark locally without a cluster manager (like YARN or Kubernetes).
   - The '*' wildcard instructs Spark to allocate one executor worker thread per
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

5. Windows Path Space Handling:
   - When project paths contain spaces (e.g. 'BDA e commerce'), Spark's internal Windows batch
     file ('find-spark-home.cmd') can misparse unquoted environment variables.
     We launch the Py4J gateway cleanly before binding Python worker paths, avoiding batch syntax errors.
"""

import os
import sys
from pathlib import Path

import pyspark
from pyspark.conf import SparkConf
from pyspark.context import SparkContext
from pyspark.java_gateway import launch_gateway
from pyspark.sql import SparkSession

# Ensure runtime directories exist
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
    project_root = Path(__file__).resolve().parent

    # 1. Prioritize Java 17 LTS if installed on the system
    java17_temurin = Path(r"C:\Program Files\Eclipse Adoptium\jdk-17.0.20.101-hotspot")
    if java17_temurin.exists():
        os.environ["JAVA_HOME"] = str(java17_temurin)
        os.environ["PATH"] = f"{java17_temurin / 'bin'};{os.environ.get('PATH', '')}"

    # 2. Fix for Windows native I/O: point to local hadoop winutils
    local_hadoop = project_root / "hadoop"
    if local_hadoop.exists():
        os.environ["HADOOP_HOME"] = str(local_hadoop)
        os.environ["PATH"] = f"{local_hadoop / 'bin'};{os.environ.get('PATH', '')}"

    # 3. Explicitly set SPARK_HOME to the pyspark package directory
    if "SPARK_HOME" not in os.environ:
        os.environ["SPARK_HOME"] = str(Path(pyspark.__file__).resolve().parent)

    # If an active SparkContext already exists, reuse it
    if SparkContext._active_spark_context is not None:
        return SparkSession.builder.getOrCreate()

    # 4. Build configuration with local optimizations
    py_exec = sys.executable
    conf = (
        SparkConf()
        .setMaster("local[*]")
        .setAppName(app_name)
        .set("spark.driver.memory", "3g")
        .set("spark.sql.shuffle.partitions", "8")
        .set("spark.default.parallelism", "8")
        .set("spark.ui.enabled", "true")
        .set("spark.ui.port", "4040")
        .set("spark.sql.adaptive.enabled", "true")
        .set("spark.sql.session.timeZone", "UTC")
        .set("spark.pyspark.python", py_exec)
        .set("spark.pyspark.driver.python", py_exec)
    )

    # 5. Clean gateway launch (prevents Windows batch scripts from tripping over path spaces)
    # When find-spark-home.cmd runs inside spark-submit.cmd, unquoted PYSPARK_PYTHON with spaces
    # causes cmd.exe to misparse 'e commerce' and output "'e' is not recognized".
    # Temporarily removing it during the batch launch avoids this entirely.
    saved_python = os.environ.pop("PYSPARK_PYTHON", None)
    saved_driver_python = os.environ.pop("PYSPARK_DRIVER_PYTHON", None)

    try:
        gateway = launch_gateway(conf)
    finally:
        # Restore PYSPARK_PYTHON so SparkContext and worker tasks execute with the virtual environment
        os.environ["PYSPARK_PYTHON"] = py_exec
        os.environ["PYSPARK_DRIVER_PYTHON"] = py_exec

    # 6. Initialize SparkContext and SparkSession
    sc = SparkContext(conf=conf, gateway=gateway)
    spark = SparkSession(sc)

    # Set log level to WARN for clean, readable output
    spark.sparkContext.setLogLevel("WARN")

    return spark


if __name__ == "__main__":
    # Quick standalone test of session creation
    print("Testing SparkSession initialization...")
    spark = get_spark("TestSession")
    print(f"Spark version: {spark.version}")
    jvm_version = spark._jvm.System.getProperty("java.version")
    jvm_vendor = spark._jvm.System.getProperty("java.vendor")
    print(f"Java Runtime: OpenJDK {jvm_version} ({jvm_vendor})")
    print(f"Master: {spark.sparkContext.master}")
    print(f"Driver Memory: {spark.conf.get('spark.driver.memory')}")
    print(f"Shuffle Partitions: {spark.conf.get('spark.sql.shuffle.partitions')}")
    print(f"Spark UI URL: {spark.sparkContext.uiWebUrl}")
    spark.stop()
    print("SparkSession stopped cleanly.")
