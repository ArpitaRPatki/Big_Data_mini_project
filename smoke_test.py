"""
Project Smoke Test & Environment Verification
---------------------------------------------
Viva Examination Notes:
1. RDD (Resilient Distributed Dataset):
   - Fundamental low-level abstraction in Apache Spark.
   - Fault-tolerant, immutable distributed collection of objects.
   - Here we demonstrate `reduceByKey`, which performs map-side combine before
     shuffling data across partitions, minimizing network I/O compared to `groupByKey`.

2. DataFrame & Spark SQL:
   - High-level tabular abstraction optimized by the Catalyst Optimizer and Tungsten Engine.
   - Columnar execution and Whole-Stage Code Generation provide 10-100x speedup over raw Python RDDs.

3. Parquet Storage Format:
   - Columnar, compressed (Snappy), and splittable binary file format.
   - Enables predicate pushdown (reading only required columns and filtering row groups before reading into memory).
"""

import time
from pathlib import Path
from pyspark.sql import functions as F
from spark_session import get_spark
from config import PARQUET_DATA_DIR


def run_rdd_word_count(spark):
    """
    Demonstrates low-level RDD transformation pipeline using reduceByKey.
    """
    print("\n--- 1. Testing RDD API (Word Count with reduceByKey) ---")
    start_time = time.time()

    sample_sentences = [
        "big data analytics with apache spark",
        "scalable machine learning and recommendation system",
        "apache spark dataframe and rdd transformations",
        "real time clickstream analytics with spark and streamlit",
    ]

    sc = spark.sparkContext
    rdd = sc.parallelize(sample_sentences, numSlices=4)

    # Word Count Pipeline:
    # 1. flatMap: split sentences into words (1 to N mapping)
    # 2. map: convert word to pair (word, 1)
    # 3. reduceByKey: combine counts per key on the worker nodes before shuffling
    counts = (
        rdd.flatMap(lambda line: line.lower().split(" "))
        .map(lambda word: (word, 1))
        .reduceByKey(lambda a, b: a + b)
        .sortBy(lambda pair: pair[1], ascending=False)
        .take(5)
    )

    elapsed = time.time() - start_time
    print(f"Top 5 frequent words: {counts}")
    print(f"RDD Word Count completed in: {elapsed:.3f} seconds.")


def run_dataframe_benchmark(spark):
    """
    Generates 1,000,000 synthetic clickstream records, performs aggregations,
    writes to columnar Parquet format, and validates row counts upon reading.
    """
    print("\n--- 2. Testing DataFrame API & Columnar Parquet I/O ---")
    num_rows = 1_000_000
    print(f"Generating synthetic clickstream dataset of {num_rows:,} records...")

    gen_start = time.time()

    # Generate synthetic clickstream events
    # Simulates: view (70%), cart (20%), purchase (10%)
    df = (
        spark.range(0, num_rows)
        .withColumn("user_id", (F.col("id") % 50_000).cast("int"))
        .withColumn("product_id", (F.col("id") % 10_000).cast("int"))
        .withColumn(
            "event_type",
            F.when(F.col("id") % 10 < 7, "view")
            .when(F.col("id") % 10 < 9, "cart")
            .otherwise("purchase"),
        )
        .withColumn("price", F.round((F.rand(seed=42) * 200.0) + 5.0, 2))
        .withColumn("category_id", (F.col("id") % 250).cast("int"))
    )

    df.cache()  # Cache DataFrame in memory for reuse
    actual_count = df.count()
    gen_time = time.time() - gen_start
    print(f"Created {actual_count:,} rows across {df.rdd.getNumPartitions()} partitions in {gen_time:.3f}s.")

    # 3. GroupBy Aggregation
    print("\nExecuting GroupBy aggregation (event distribution & average price):")
    agg_start = time.time()
    agg_df = (
        df.groupBy("event_type")
        .agg(
            F.count("*").alias("total_events"),
            F.round(F.avg("price"), 2).alias("avg_price"),
            F.round(F.sum("price"), 2).alias("total_value"),
        )
        .orderBy(F.desc("total_events"))
    )
    agg_df.show()
    agg_time = time.time() - agg_start
    print(f"Aggregation completed in: {agg_time:.3f} seconds.")

    # 4. Write to Parquet
    parquet_target = PARQUET_DATA_DIR / "smoke_test_clickstream.parquet"
    print(f"\nWriting dataset to Parquet at: {parquet_target} ...")
    write_start = time.time()
    df.write.mode("overwrite").parquet(str(parquet_target))
    write_time = time.time() - write_start
    print(f"Parquet write completed in: {write_time:.3f} seconds.")

    # 5. Read from Parquet
    print("\nReading dataset back from Parquet...")
    read_start = time.time()
    df_read = spark.read.parquet(str(parquet_target))
    read_count = df_read.count()
    read_time = time.time() - read_start
    print(f"Parquet read verified: {read_count:,} rows read back in {read_time:.3f} seconds.")

    # Validation assertion
    assert actual_count == read_count, f"Row count mismatch! Expected {actual_count}, got {read_count}"
    print("\n[SUCCESS] Parquet persistence and read-back validation passed!")


def main():
    print("=" * 65)
    print("APACHE SPARK SMOKE TEST & ENVIRONMENT VERIFICATION")
    print("=" * 65)

    overall_start = time.time()
    spark = get_spark("SmokeTest")

    print(f"Spark Version: {spark.version}")
    jvm_ver = spark._jvm.System.getProperty("java.version")
    jvm_vendor = spark._jvm.System.getProperty("java.vendor")
    print(f"Java Runtime: OpenJDK {jvm_ver} ({jvm_vendor})")
    print(f"Driver Master: {spark.sparkContext.master}")
    print(f"Shuffle Partitions: {spark.conf.get('spark.sql.shuffle.partitions')}")
    print(f"Driver Memory: {spark.conf.get('spark.driver.memory')}")

    try:
        run_rdd_word_count(spark)
        run_dataframe_benchmark(spark)
    finally:
        spark.stop()
        print("\nSparkSession closed cleanly.")

    total_time = time.time() - overall_start
    print(f"\nTotal Smoke Test Execution Time: {total_time:.2f} seconds.")
    print("=" * 65)


if __name__ == "__main__":
    main()
