# Scalable Customer Behavior Analytics and Product Recommendation for an Online Retailer Using Apache Spark

## Project Overview
This Big Data Analytics project builds an end-to-end, high-performance distributed data pipeline for analyzing tens of millions of e-commerce clickstream events from the REES46 multi-category online store dataset. Leveraging **Apache Spark (PySpark)** for scalable distributed batch and stream-style processing, **Spark MLlib** for machine learning recommendations (Alternating Least Squares collaborative filtering), and **Streamlit** for interactive analytical dashboards, the system extracts real-time customer behavior patterns, conversion funnels, and personalized product recommendations at scale.

---

## Prerequisites & System Environment
- **Operating System:** Windows 11 (64-bit) / Linux / macOS
- **CPU:** Multi-core processor (Configured with `local[*]` to leverage all cores/threads)
- **RAM:** Minimum 8 GB (Spark driver allocated 3 GB, leaving 5 GB for the OS)
- **Java:** OpenJDK 8 or 17 LTS (Required for Apache Spark execution engine)
- **Python:** Python 3.11.x (Using dedicated virtual environment `.venv`)

---

## Quick Setup Instructions

### 1. Clone or Open the Repository
```bash
cd "d:/BDA e commerce"
```

### 2. Activate the Virtual Environment
- **On Windows (PowerShell):**
  ```powershell
  .\.venv\Scripts\Activate.ps1
  ```
- **On Windows (Command Prompt):**
  ```cmd
  .venv\Scripts\activate.bat
  ```
- **On Linux / macOS:**
  ```bash
  source .venv/bin/activate
  ```

### 3. Install Required Dependencies
All dependencies are pinned for stability in `requirements.txt`:
```bash
pip install -r requirements.txt
```

---

## Running the Smoke Test
The smoke test verifies that Java, PySpark, RDD distributed primitives (`reduceByKey`), DataFrame catalyst optimizations, and local columnar Parquet serialization are functioning correctly:

```powershell
python smoke_test.py
```

### Expected Output
1. Spark session initialized with `local[*]` and `spark.sql.shuffle.partitions=8`.
2. Execution of RDD Word Count displaying top words.
3. Generation of 1,000,000 synthetic clickstream events.
4. Aggregation by `event_type` showing event counts and average prices.
5. Successful write to Parquet at `data/parquet/smoke_test_clickstream.parquet` and complete row count verification upon reading back.

---

## Project Structure
```text
bigdata-ecommerce-analytics/
├── data/
│   ├── raw/             # Raw clickstream CSV files
│   ├── parquet/         # High-performance columnar Parquet files
│   └── output/          # Exported metrics and recommendation artifacts
├── ingest/              # Data ingestion and ingestion benchmark scripts
├── clean/               # Data cleaning, schema enforcement, and deduplication
├── analysis/            # Funnel analysis, RFM segmentation, and EDA
├── ml/                  # ALS collaborative filtering and recommendation models
├── optimization/        # Partition tuning, caching, and execution profiling
├── dashboard/           # Interactive Streamlit analytics web application
├── report/              # Generated visualizations, plots, and summary reports
├── logs/                # Spark execution logs
├── config.py            # Centralized pathlib directory paths
├── spark_session.py     # Local-optimized SparkSession builder
├── smoke_test.py        # Environment verification and pipeline benchmark
├── requirements.txt     # Pinned Python package dependencies
├── .gitignore           # Git ignore configuration
└── README.md            # Project documentation and setup guide
```

---

## Windows Note: Hadoop WinUtils & Native IO
Apache Spark is built on Hadoop's FileSystem API, which expects POSIX-style permissions. On Windows, Hadoop requires `winutils.exe` and `hadoop.dll` under `%HADOOP_HOME%\bin` for native file access and permissions management. 
- If running on Windows, the project includes an automated setup in `hadoop/bin/` so Parquet files write without requiring complex manual OS configuration.
- Alternatively, running inside **WSL2 (Windows Subsystem for Linux)** provides a native Linux environment where winutils is unnecessary.
