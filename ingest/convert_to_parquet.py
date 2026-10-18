"""
Raw Clickstream Ingestion & Parquet Conversion Module
-----------------------------------------------------
Viva Examination Notes:
1. Explicit Schema vs inferSchema:
   - With a ~9 GB raw CSV file (~67M rows), setting `inferSchema=True` forces
     Spark to read the entire dataset twice: once to infer data types and once to load it.
   - Defining an explicit `StructType` schema avoids the initial full-scan overhead,
     drastically reducing ingestion time and preventing memory exhaustion.

2. Raw Layer Schema Strategy (Bronze Layer):
   - In modern lakehouse architectures (Medallion architecture: Bronze -> Silver -> Gold),
     the raw ingestion layer captures data as faithfully as possible without loss.
   - `event_time` is loaded as `StringType` to prevent silent timezone shifts or
     parsing drops during ingestion; standardized timestamp casting happens in the clean layer.
   - Identifiers and counts (`product_id`, `category_id`, `user_id`) are mapped to `LongType`
     (64-bit integer) to safely accommodate massive eCommerce ID ranges.

3. Snappy-Compressed Columnar Parquet:
   - Parquet organizes data by column rather than row, enabling high-ratio compression (Snappy)
     and allowing downstream analytics to read only the columns needed (column projection)
     while skipping non-matching data blocks (predicate pushdown).
"""

import argparse
import sys
import time
from pathlib import Path

# Add project root to path so config and spark_session are readily accessible
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pyspark.sql.types import (
    DoubleType,
    LongType,
    StringType,
    StructField,
    StructType,
)

from config import PARQUET_DATA_DIR, RAW_DATA_DIR
from spark_session import get_spark

# -----------------------------------------------------------------------------
# Explicit PySpark Schema for REES46 eCommerce Clickstream Dataset
# -----------------------------------------------------------------------------
CLICKSTREAM_SCHEMA = StructType([
    StructField("event_time", StringType(), nullable=True),      # Raw timestamp string (e.g., '2019-11-01 00:00:00 UTC')
    StructField("event_type", StringType(), nullable=True),      # Action: 'view', 'cart', 'purchase'
    StructField("product_id", LongType(), nullable=True),        # Product identifier
    StructField("category_id", LongType(), nullable=True),       # Category hierarchy identifier (64-bit int)
    StructField("category_code", StringType(), nullable=True),    # Category taxonomy string (e.g., 'electronics.smartphone')
    StructField("brand", StringType(), nullable=True),            # Manufacturer / Brand name
    StructField("price", DoubleType(), nullable=True),            # Product price in USD
    StructField("user_id", LongType(), nullable=True),           # Unique user identifier
    StructField("user_session", StringType(), nullable=True),     # Unique user session UUID token
])


def format_bytes(size_bytes: int) -> str:
    """Formats raw byte counts into human-readable strings (KB, MB, GB)."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"


def get_dir_size(directory: Path) -> int:
    """Calculates total disk usage in bytes for a directory."""
    if not directory.exists():
        return 0
    return sum(f.stat().st_size for f in directory.rglob("*") if f.is_file())


def validate_csv_header(csv_path: Path, expected_schema: StructType) -> None:
    """
    Validates that the physical CSV header matches the expected schema column names.
    Raises ValueError if headers do not align, preventing malformed ingestion.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Source CSV file not found at: {csv_path}")

    with open(csv_path, "r", encoding="utf-8") as f:
        header_line = f.readline().strip()

    actual_columns = [col.strip() for col in header_line.split(",")]
    expected_columns = [field.name for field in expected_schema.fields]

    if actual_columns != expected_columns:
        raise ValueError(
            f"CSV Header Mismatch in {csv_path.name}!\n"
            f"Expected: {expected_columns}\n"
            f"Found:    {actual_columns}"
        )


def ingest_clickstream_to_parquet(
    input_file: Path,
    limit: int = None,
    overwrite: bool = False,
) -> Path:
    """
    Reads the raw REES46 CSV, applies explicit schema, converts to Snappy Parquet,
    and performs read-back schema and row-count validation.

    Parameters:
    -----------
    input_file : Path
        Path to the source CSV file in data/raw/.
    limit : int, optional
        Maximum number of rows to ingest (for rapid testing/sampling). None for full dataset.
    overwrite : bool, default False
        Whether to overwrite an existing Parquet output folder.

    Returns:
    --------
    Path
        Destination path of the written Parquet directory.
    """
    print("=" * 75)
    print("CHUNK 2: REES46 E-COMMERCE DATASET INGESTION & PARQUET CONVERSION")
    print("=" * 75)

    start_time = time.time()

    # 1. Validate Input File & Headers
    print(f"Source CSV:       {input_file.resolve()}")
    if not input_file.exists():
        raise FileNotFoundError(f"Input dataset does not exist: {input_file}")

    file_size_bytes = input_file.stat().st_size
    print(f"Source File Size: {format_bytes(file_size_bytes)} ({file_size_bytes:,} bytes)")

    print("\nValidating CSV header against expected schema...")
    validate_csv_header(input_file, CLICKSTREAM_SCHEMA)
    print("[OK] Column headers match expected REES46 schema perfectly.")

    # 2. Determine Output Directory Name
    # Clearly denotes whether output contains a sample or the complete input
    if limit is not None and limit > 0:
        output_dir_name = f"raw_clickstream_sample_{limit}.parquet"
        is_sample = True
    else:
        output_dir_name = "raw_clickstream_full.parquet"
        is_sample = False

    output_path = PARQUET_DATA_DIR / output_dir_name

    print(f"\nTarget Parquet:   {output_path.resolve()}")
    print(f"Execution Mode:   {'SAMPLE (' + f'{limit:,}' + ' rows)' if is_sample else 'FULL DATASET'}")

    # Check for existing output
    if output_path.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output directory already exists at: {output_path}\n"
                f"To replace it, pass --overwrite to the command."
            )
        else:
            print("[WARN] Existing output directory will be overwritten (--overwrite specified).")

    # 3. Initialize Spark Session
    app_title = f"IngestClickstream_{'Sample' if is_sample else 'Full'}"
    spark = get_spark(app_title)

    try:
        # Display explicit schema for viva review
        print("\n--- Explicit PySpark Ingestion Schema ---")
        for field in CLICKSTREAM_SCHEMA.fields:
            print(f"  - {field.name:15s} : {field.dataType.simpleString():12s} (nullable={field.nullable})")

        # 4. Read Raw CSV with Explicit Schema (Strictly avoiding inferSchema)
        print(f"\nReading raw CSV data from {input_file.name}...")
        read_start = time.time()
        df = (
            spark.read.format("csv")
            .option("header", "true")
            .option("delimiter", ",")
            .option("mode", "PERMISSIVE")            # Handles malformed records gracefully
            .schema(CLICKSTREAM_SCHEMA)
            .load(str(input_file))
        )

        # Apply row limit if running a sample
        if is_sample:
            df = df.limit(limit)

        # Count rows selected for writing
        rows_to_write = df.count()
        read_duration = time.time() - read_start
        print(f"Identified {rows_to_write:,} records to ingest in {read_duration:.2f} seconds.")

        # 5. Write to Columnar Parquet with Snappy Compression
        print(f"\nWriting dataset to Snappy Parquet at: {output_path} ...")
        write_start = time.time()
        (
            df.write
            .mode("overwrite" if overwrite else "error")
            .option("compression", "snappy")
            .parquet(str(output_path))
        )
        write_duration = time.time() - write_start
        output_size_bytes = get_dir_size(output_path)

        print(f"Parquet write completed in: {write_duration:.2f} seconds.")
        print(f"Output Directory Size: {format_bytes(output_size_bytes)} ({output_size_bytes:,} bytes)")
        if not is_sample:
            ratio = (1.0 - (output_size_bytes / file_size_bytes)) * 100.0
            print(f"Storage Space Saved via Parquet Compression: {ratio:.1f}%")

        # 6. Read Back and Verify Schema & Row Count (Requirement 9)
        print("\n--- Verification: Validating Written Parquet ---")
        verify_start = time.time()
        df_verified = spark.read.parquet(str(output_path))

        # Check Schema Equality
        verified_cols = [f.name for f in df_verified.schema.fields]
        expected_cols = [f.name for f in CLICKSTREAM_SCHEMA.fields]
        assert verified_cols == expected_cols, (
            f"Schema mismatch on read-back! Expected {expected_cols}, got {verified_cols}"
        )
        print(f"[OK] Schema verified: all {len(verified_cols)} columns match expected types.")

        # Check Row Count Equality
        verified_count = df_verified.count()
        verify_duration = time.time() - verify_start
        print(f"[OK] Row count verified: {verified_count:,} rows match write selection exactly.")
        assert verified_count == rows_to_write, (
            f"Row count mismatch! Written: {rows_to_write}, Read: {verified_count}"
        )

        # Display Sample Rows
        print(f"\nSample of Ingested Records (5 rows):")
        df_verified.show(5, truncate=False)

    finally:
        spark.stop()
        print("SparkSession stopped cleanly.")

    total_time = time.time() - start_time
    print("=" * 75)
    print(f"Ingestion Finished Successfully in {total_time:.2f} seconds.")
    print("=" * 75)
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Ingest REES46 eCommerce CSV to columnar Snappy Parquet using PySpark."
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        default=RAW_DATA_DIR / "2019-Nov.csv",
        help="Path to the raw CSV file (defaults to data/raw/2019-Nov.csv)",
    )
    parser.add_argument(
        "--sample-rows",
        "--limit",
        dest="limit",
        type=int,
        default=None,
        help="Number of rows to ingest for a quick sample run (e.g. 100000). Omit for full dataset.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Overwrite existing Parquet output directory if it already exists.",
    )

    args = parser.parse_args()

    try:
        ingest_clickstream_to_parquet(
            input_file=args.input_file,
            limit=args.limit,
            overwrite=args.overwrite,
        )
    except Exception as e:
        print(f"\n[ERROR] Ingestion failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
