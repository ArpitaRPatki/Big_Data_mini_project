"""
Central Configuration Module
----------------------------
Viva Explanation:
This module centralizes all directory paths using Python's standard `pathlib.Path`.
Centralizing paths prevents hardcoding across scripts, ensuring portability across
different environments (Windows, Linux, macOS) and preventing file-not-found errors.
"""

from pathlib import Path

# Project root directory (the directory containing config.py)
BASE_DIR: Path = Path(__file__).resolve().parent

# Main Data directories
DATA_DIR: Path = BASE_DIR / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PARQUET_DATA_DIR: Path = DATA_DIR / "parquet"
OUTPUT_DATA_DIR: Path = DATA_DIR / "output"

# Logs directory for tracking Spark jobs and pipeline runs
LOGS_DIR: Path = BASE_DIR / "logs"

# Project module directories
INGEST_DIR: Path = BASE_DIR / "ingest"
CLEAN_DIR: Path = BASE_DIR / "clean"
ANALYSIS_DIR: Path = BASE_DIR / "analysis"
ML_DIR: Path = BASE_DIR / "ml"
OPTIMIZATION_DIR: Path = BASE_DIR / "optimization"
DASHBOARD_DIR: Path = BASE_DIR / "dashboard"
REPORT_DIR: Path = BASE_DIR / "report"

# Ensure runtime directories exist
for path in [RAW_DATA_DIR, PARQUET_DATA_DIR, OUTPUT_DATA_DIR, LOGS_DIR]:
    path.mkdir(parents=True, exist_ok=True)
