"""
Data Cleaning & Feature Engineering Module (Chunk 3)
----------------------------------------------------
Viva Examination Notes:
1. Bronze to Silver Lakehouse Transformation:
   - In Chunk 2 (Bronze layer), raw data was ingested as-is with string timestamps
     to ensure zero data loss during extraction.
   - In Chunk 3 (Silver layer), we apply data quality filters, schema normalization,
     and derive analytical features essential for downstream analytics and ML.

2. Transparent Data Quality (No Silent Dropping):
   - Rather than silently discarding rows, every cleaning filter is quantified:
     exact counts of duplicates, negative prices, and invalid event types are logged
     and saved to an audit report in data/output/.

3. Business Logic for Zero Prices:
   - In eCommerce clickstream data, price = $0.00 corresponds to promotional giveaways,
     complimentary gifts, or zero-cost service interactions.
   - We explicitly retain valid zero prices in the '0_Free' price band while filtering
     out unphysical negative prices (price < 0).

4. Missing Value Imputation Strategy:
   - High cardinality categorical fields like 'brand' (~15.8% missing) and 'category_code'
     (~33.8% missing) cannot be dropped without discarding massive volumes of legitimate user sessions.
   - We impute missing values with 'Unknown' / 'unknown' and engineer boolean indicator flags
     ('is_brand_missing', 'is_category_missing') so downstream ML models can learn whether
     missing metadata correlates with user purchase propensity.

5. Engineered Temporal & Price Features:
   - 'hour_of_day' (0-23): Captures diurnal purchasing cycles.
   - 'day_of_week' (1-7): Captures weekend vs weekday shopping behavior.
   - 'day_name': Human-readable weekday string for dashboard visualization.
   - 'is_weekend': Binary indicator (1 = Sat/Sun, 0 = Mon-Fri).
   - 'price_band': Stratified pricing tiers for segmentation and recommendations.
   - 'category_root': High-level product category (e.g., 'electronics' from 'electronics.smartphone').
"""

import argparse
import json
import sys
import time
from pathlib import Path

# Add project root to path for config and spark_session imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pyspark.sql import functions as F
from pyspark.sql import DataFrame
from pyspark.sql.types import IntegerType, TimestampType

from config import OUTPUT_DATA_DIR, PARQUET_DATA_DIR
from spark_session import get_spark


def profile_raw_data(df: DataFrame) -> dict:
    """
    Computes baseline quality metrics on the uncleaned dataset before applying rules.
    """
    total_records = df.count()

    # 1. Null / empty counts per column
    null_exprs = [
        F.count(
            F.when(F.col(c).isNull() | (F.trim(F.col(c)) == ""), c)
        ).alias(c)
        for c in df.columns
    ]
    null_counts_row = df.select(null_exprs).collect()[0].asDict()

    # 2. Duplicate rows
    distinct_count = df.dropDuplicates().count()
    duplicate_count = total_records - distinct_count

    # 3. Timestamp validity
    parsed_ts = F.to_timestamp(F.col("event_time"), "yyyy-MM-dd HH:mm:ss 'UTC'")
    invalid_ts_count = df.filter(parsed_ts.isNull()).count()

    # 4. Price anomalies
    neg_price_count = df.filter(F.col("price") < 0.0).count()
    zero_price_count = df.filter(F.col("price") == 0.0).count()
    null_price_count = df.filter(F.col("price").isNull()).count()

    # 5. Event type distribution
    event_distribution = {
        row["event_type"]: row["count"]
        for row in df.groupBy("event_type").count().collect()
    }

    return {
        "total_records": total_records,
        "duplicate_records": duplicate_count,
        "invalid_timestamps": invalid_ts_count,
        "negative_prices": neg_price_count,
        "zero_prices": zero_price_count,
        "null_prices": null_price_count,
        "column_null_counts": null_counts_row,
        "event_distribution": event_distribution,
    }


def clean_and_engineer_features(
    input_path: Path,
    output_path: Path = None,
    overwrite: bool = False,
) -> tuple[Path, dict]:
    """
    Executes the full cleaning and feature engineering pipeline on Parquet clickstream data.
    """
    print("=" * 75)
    print("CHUNK 3: DATA CLEANING & FEATURE ENGINEERING")
    print("=" * 75)

    start_time = time.time()

    # Verify input exists
    if not input_path.exists():
        raise FileNotFoundError(f"Input Parquet dataset not found at: {input_path}")

    # Determine default output directory name based on input
    if output_path is None:
        input_name = input_path.name
        if "sample" in input_name:
            output_name = input_name.replace("raw_clickstream", "cleaned_clickstream")
        else:
            output_name = "cleaned_clickstream_full.parquet"
        output_path = PARQUET_DATA_DIR / output_name

    print(f"Input Parquet:    {input_path.resolve()}")
    print(f"Output Parquet:   {output_path.resolve()}")

    # Check for existing output
    if output_path.exists():
        if not overwrite:
            raise FileExistsError(
                f"Target output directory '{output_path}' already exists.\n"
                f"Pass --overwrite to replace it."
            )
        else:
            print("[WARN] Existing output directory will be overwritten (--overwrite specified).")

    # Initialize Spark
    spark = get_spark("DataCleaningAndFeatureEngineering")

    try:
        # 1. Load Input Parquet
        print(f"\nLoading source Parquet from {input_path.name}...")
        df_raw = spark.read.parquet(str(input_path))
        initial_count = df_raw.count()
        print(f"[OK] Successfully loaded {initial_count:,} records.")

        # 2. Baseline Profiling
        print("\n--- 1. Baseline Data Quality Profiling ---")
        metrics = profile_raw_data(df_raw)

        print(f"Total Input Rows:       {metrics['total_records']:,}")
        print(f"Exact Duplicate Rows:   {metrics['duplicate_records']:,} ({metrics['duplicate_records']/initial_count*100:.2f}%)")
        print(f"Invalid Timestamps:     {metrics['invalid_timestamps']:,}")
        print(f"Negative Prices (< $0): {metrics['negative_prices']:,}")
        print(f"Zero Prices ($0.00):    {metrics['zero_prices']:,} (Preserved as valid promotional interactions)")
        print(f"Event Types Detected:   {metrics['event_distribution']}")

        print("\nMissing / Blank Values by Column:")
        for col_name, null_cnt in metrics["column_null_counts"].items():
            pct = (null_cnt / initial_count) * 100.0
            print(f"  - {col_name:15s}: {null_cnt:7,d} ({pct:5.2f}%)")

        # 3. Document Cleaning Decisions & Rule Execution
        print("\n--- 2. Applying Explicit Cleaning Rules ---")
        cleaning_audit = {
            "input_records": initial_count,
            "rules_applied": {},
        }

        # Rule 1: Remove exact duplicate rows
        # In telemetry clickstream data, multi-clicks or logger retries produce identical duplicate events
        df_dedup = df_raw.dropDuplicates()
        dedup_count = df_dedup.count()
        dupes_removed = initial_count - dedup_count
        cleaning_audit["rules_applied"]["rule_1_drop_exact_duplicates"] = {
            "description": "Remove completely identical event log entries across all columns",
            "records_affected": dupes_removed,
            "percentage_of_input": round((dupes_removed / initial_count) * 100.0, 3),
        }
        print(f"Rule 1 [Drop Duplicates]: Removed {dupes_removed:,} exact duplicate records.")

        # Rule 2: Filter negative prices (corrupt sensor/pricing data)
        # Note: Valid zero prices are explicitly PRESERVED as per project requirements
        df_valid_price = df_dedup.filter((F.col("price") >= 0.0) | F.col("price").isNull())
        price_filtered_count = df_valid_price.count()
        neg_prices_removed = dedup_count - price_filtered_count
        cleaning_audit["rules_applied"]["rule_2_filter_negative_prices"] = {
            "description": "Filter unphysical negative prices while preserving valid zero prices",
            "records_affected": neg_prices_removed,
            "percentage_of_input": round((neg_prices_removed / initial_count) * 100.0, 3),
        }
        print(f"Rule 2 [Negative Prices]: Filtered {neg_prices_removed:,} negative-price records.")

        # Rule 3: Validate event_type belongs to expected eCommerce actions
        valid_events = ["view", "cart", "purchase"]
        df_valid_events = df_valid_price.filter(F.col("event_type").isin(valid_events))
        valid_events_count = df_valid_events.count()
        invalid_events_removed = price_filtered_count - valid_events_count
        cleaning_audit["rules_applied"]["rule_3_valid_event_types"] = {
            "description": "Ensure event_type is one of ('view', 'cart', 'purchase')",
            "records_affected": invalid_events_removed,
            "percentage_of_input": round((invalid_events_removed / initial_count) * 100.0, 3),
        }
        print(f"Rule 3 [Valid Event Types]: Filtered {invalid_events_removed:,} unexpected event records.")

        # 4. Feature Engineering & Normalization
        print("\n--- 3. Feature Engineering & Categorical Normalization ---")

        # Step 4A: Parse event_time into UTC Timestamp
        df_features = df_valid_events.withColumn(
            "event_timestamp",
            F.to_timestamp(F.col("event_time"), "yyyy-MM-dd HH:mm:ss 'UTC'")
        )

        # Rule 4: Filter any records where timestamp was unparseable
        df_valid_ts = df_features.filter(F.col("event_timestamp").isNotNull())
        valid_ts_count = df_valid_ts.count()
        invalid_ts_removed = valid_events_count - valid_ts_count
        cleaning_audit["rules_applied"]["rule_4_valid_timestamp"] = {
            "description": "Parse event_time to TimestampType and remove unparseable timestamps",
            "records_affected": invalid_ts_removed,
            "percentage_of_input": round((invalid_ts_removed / initial_count) * 100.0, 3),
        }
        print(f"Rule 4 [Timestamp Parsing]: Filtered {invalid_ts_removed:,} unparseable timestamp records.")

        # Step 4B: Temporal Feature Extraction
        # hour_of_day: 0 to 23
        # day_of_week: 1 (Sunday) to 7 (Saturday)
        # day_name: e.g., 'Friday'
        # is_weekend: 1 if Saturday or Sunday, else 0
        df_engineered = (
            df_valid_ts
            .withColumn("hour_of_day", F.hour(F.col("event_timestamp")).cast(IntegerType()))
            .withColumn("day_of_week", F.dayofweek(F.col("event_timestamp")).cast(IntegerType()))
            .withColumn("day_name", F.date_format(F.col("event_timestamp"), "EEEE"))
            .withColumn(
                "is_weekend",
                F.when(F.col("day_of_week").isin(1, 7), 1).otherwise(0).cast(IntegerType())
            )
        )

        # Step 4C: Stratified Price Band Feature
        # 0_Free: $0.00
        # 1_Budget: $0.01 - $25.00
        # 2_Affordable: $25.01 - $100.00
        # 3_MidRange: $100.01 - $500.00
        # 4_Premium: > $500.00
        df_engineered = df_engineered.withColumn(
            "price_band",
            F.when(F.col("price") == 0.0, "0_Free")
            .when((F.col("price") > 0.0) & (F.col("price") <= 25.0), "1_Budget")
            .when((F.col("price") > 25.0) & (F.col("price") <= 100.0), "2_Affordable")
            .when((F.col("price") > 100.0) & (F.col("price") <= 500.0), "3_MidRange")
            .otherwise("4_Premium")
        )

        # Step 4D: Normalization of Missing Categoricals & Metadata Flags
        # Normalize missing brand to 'Unknown' and missing category_code to 'unknown'
        # Preserve is_brand_missing and is_category_missing indicators
        df_final = (
            df_engineered
            .withColumn(
                "is_brand_missing",
                F.when(F.col("brand").isNull() | (F.trim(F.col("brand")) == ""), True).otherwise(False)
            )
            .withColumn(
                "brand",
                F.when(F.col("is_brand_missing"), "Unknown").otherwise(F.trim(F.col("brand")))
            )
            .withColumn(
                "is_category_missing",
                F.when(F.col("category_code").isNull() | (F.trim(F.col("category_code")) == ""), True).otherwise(False)
            )
            .withColumn(
                "category_code",
                F.when(F.col("is_category_missing"), "unknown").otherwise(F.trim(F.col("category_code")))
            )
            .withColumn(
                "category_root",
                F.when(
                    F.col("category_code") == "unknown", "unknown"
                ).otherwise(F.split(F.col("category_code"), r"\.")[0])
            )
        )

        final_count = df_final.count()
        total_removed = initial_count - final_count
        cleaning_audit["output_records"] = final_count
        cleaning_audit["total_records_removed"] = total_removed
        cleaning_audit["retention_rate_pct"] = round((final_count / initial_count) * 100.0, 3)

        print(f"\nFinal Cleaned Records:  {final_count:,}")
        print(f"Total Records Removed:  {total_removed:,} ({total_removed/initial_count*100:.2f}%)")
        print(f"Pipeline Retention Rate: {cleaning_audit['retention_rate_pct']}%")

        # 5. Write to Snappy Parquet
        print(f"\nWriting cleaned dataset to Snappy Parquet at: {output_path} ...")
        write_start = time.time()
        (
            df_final.write
            .mode("overwrite" if overwrite else "error")
            .option("compression", "snappy")
            .parquet(str(output_path))
        )
        write_time = time.time() - write_start
        print(f"Parquet write finished in: {write_time:.2f} seconds.")

        # 6. Read Back and Validate Schema & Row Count
        print("\n--- 4. Read-Back Output Validation ---")
        df_verified = spark.read.parquet(str(output_path))
        verified_count = df_verified.count()
        assert verified_count == final_count, (
            f"Row count mismatch! Expected {final_count}, got {verified_count}"
        )
        print(f"[OK] Output row count verified: {verified_count:,} records.")

        # Verify added feature columns
        expected_new_cols = [
            "event_timestamp", "hour_of_day", "day_of_week", "day_name",
            "is_weekend", "price_band", "is_brand_missing", "is_category_missing", "category_root"
        ]
        actual_output_cols = df_verified.columns
        for c in expected_new_cols:
            assert c in actual_output_cols, f"Expected engineered column '{c}' missing from output!"
        print(f"[OK] All {len(expected_new_cols)} engineered feature columns present and validated.")

        # Display Sample Rows
        print("\nSample Cleaned & Engineered Records (5 rows):")
        df_verified.select(
            "event_timestamp", "event_type", "price", "price_band",
            "hour_of_day", "day_name", "brand", "category_root"
        ).show(5, truncate=False)

        # 7. Price Band Distribution Review
        print("\nPrice Band Distribution:")
        df_verified.groupBy("price_band").count().orderBy("price_band").show()

        # 8. Save Audit Summary Report
        summary_filename = f"cleaning_summary_{output_path.stem}.json"
        summary_file = OUTPUT_DATA_DIR / summary_filename
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(cleaning_audit, f, indent=4)
        print(f"[OK] Cleaning audit summary report saved to: {summary_file}")

    finally:
        spark.stop()
        print("\nSparkSession closed cleanly.")

    total_time = time.time() - start_time
    print("=" * 75)
    print(f"Chunk 3 Clean & Feature Pipeline Completed in {total_time:.2f} seconds.")
    print("=" * 75)

    return output_path, cleaning_audit


def main():
    parser = argparse.ArgumentParser(
        description="Clean clickstream Parquet dataset and engineer temporal/pricing features."
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=PARQUET_DATA_DIR / "raw_clickstream_sample_100000.parquet",
        help="Path to raw Parquet input (default: data/parquet/raw_clickstream_sample_100000.parquet)",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Optional custom destination path for cleaned Parquet dataset.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Overwrite existing output directory if it already exists.",
    )

    args = parser.parse_args()

    try:
        clean_and_engineer_features(
            input_path=args.input_path,
            output_path=args.output_path,
            overwrite=args.overwrite,
        )
    except Exception as e:
        print(f"\n[ERROR] Cleaning pipeline failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
