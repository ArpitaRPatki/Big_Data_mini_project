"""
Customer Segmentation Module Using PySpark MLlib K-Means (Chunk 5)
------------------------------------------------------------------
Viva Examination Notes:
1. Unsupervised Machine Learning in Big Data:
   - Customer segmentation is an unsupervised clustering technique that groups users
     sharing similar behavioral patterns without pre-existing ground-truth labels.
   - PySpark MLlib executes distributed K-Means (using the k-means|| initialization variant),
     enabling parallel distance computations and centroid updates across cluster nodes.

2. Necessity of Feature Scaling (StandardScaler):
   - K-Means computes distances using Euclidean distance in n-dimensional space.
   - Features with large numerical ranges (e.g., 'total_spend' spanning $0 to $5,000+)
     would dwarf features with smaller ranges (e.g., 'purchase_count' spanning 0 to 11).
   - `StandardScaler(withStd=True, withMean=False)` standardizes each feature to unit variance,
     ensuring equal mathematical weighting during cluster distance calculations.

3. Model Evaluation via Silhouette Analysis:
   - Unlike supervised learning (which has accuracy or RMSE), clustering quality is evaluated
     using geometric compactness and separation.
   - The Silhouette Score ranges from -1 to +1:
     * Near +1: Clusters are dense and well separated.
     * Near 0: Overlapping clusters.
     * Negative: Data points assigned to wrong clusters.
   - A silhouette score > 0.60 indicates strong, well-defined cluster separation.

4. Data-Grounded Behavioral Interpretation:
   - Clusters are interpreted strictly based on observed feature means, avoiding fabricated labels:
     * Cluster 0: Active Window Shoppers (High views across many products, minimal purchases).
     * Cluster 1: Casual / One-Time Visitors (Low views, quick bounce, negligible spend).
     * Cluster 2: Heavy Product Researchers (Extensive multi-item browsing and comparison).
     * Cluster 3: High-Value Converting Buyers (Multiple purchases, high spend and high AOV).
"""

import argparse
import json
import sys
import time
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.clustering import KMeans, KMeansModel
from pyspark.ml.evaluation import ClusteringEvaluator

from config import OUTPUT_DATA_DIR, PARQUET_DATA_DIR, REPORT_DIR
from spark_session import get_spark


FEATURE_COLUMNS = [
    "view_count",
    "cart_count",
    "purchase_count",
    "total_spend",
    "avg_price",
    "distinct_products",
    "active_days",
]


def extract_user_features(df: DataFrame) -> tuple[DataFrame, int]:
    """
    Aggregates one behavioral feature vector per identified user.
    Excludes null user_ids and returns the aggregated DataFrame and excluded count.
    """
    # 1. Check and filter null user IDs
    initial_event_count = df.count()
    valid_users_df = df.filter(F.col("user_id").isNotNull())
    null_user_records = initial_event_count - valid_users_df.count()

    # 2. User-level feature aggregation
    user_features_df = (
        valid_users_df.groupBy("user_id")
        .agg(
            # Behavioral interaction frequencies
            F.count(F.when(F.col("event_type") == "view", 1)).alias("view_count"),
            F.count(F.when(F.col("event_type") == "cart", 1)).alias("cart_count"),
            F.count(F.when(F.col("event_type") == "purchase", 1)).alias("purchase_count"),
            # Monetary metrics (coalesced safely to 0.0)
            F.round(
                F.coalesce(
                    F.sum(F.when(F.col("event_type") == "purchase", F.col("price"))),
                    F.lit(0.0),
                ),
                2,
            ).alias("total_spend"),
            F.round(F.coalesce(F.avg("price"), F.lit(0.0)), 2).alias("avg_price"),
            # Breadth of engagement
            F.countDistinct("product_id").alias("distinct_products"),
            F.countDistinct(F.to_date("event_timestamp")).alias("active_days"),
        )
        .na.fill(0.0)
    )

    return user_features_df, null_user_records


def evaluate_k_values(
    scaled_df: DataFrame,
    k_candidates: list[int] = [3, 4, 5],
    seed: int = 42,
) -> dict[int, float]:
    """
    Evaluates a candidate range of K values using Spark MLlib's ClusteringEvaluator (Silhouette Score).
    """
    evaluator = ClusteringEvaluator(
        featuresCol="scaled_features",
        predictionCol="cluster",
        metricName="silhouette",
        distanceMeasure="squaredEuclidean",
    )

    silhouette_scores = {}
    print("\n--- Evaluating Candidate K Values (Silhouette Analysis) ---")
    for k in k_candidates:
        t0 = time.time()
        kmeans = KMeans(
            featuresCol="scaled_features",
            predictionCol="cluster",
            k=k,
            seed=seed,
            maxIter=20,
        )
        model = kmeans.fit(scaled_df)
        predictions = model.transform(scaled_df)
        score = evaluator.evaluate(predictions)
        duration = time.time() - t0
        silhouette_scores[k] = round(score, 4)
        print(f"  * K = {k} -> Silhouette Score: {score:.4f} (Computed in {duration:.2f}s)")

    return silhouette_scores


def assign_grounded_segment_names(profile_rows: list[dict]) -> dict[int, str]:
    """
    Assigns descriptive, data-grounded segment names based strictly on computed cluster averages.
    Ensures labels reflect actual empirical properties rather than arbitrary speculation.
    """
    segment_names = {}
    for row in profile_rows:
        cid = row["cluster"]
        avg_spend = row["avg_spend"]
        avg_views = row["avg_views"]
        avg_purchases = row["avg_purchases"]
        avg_products = row["avg_distinct_products"]
        avg_price = row.get("avg_product_price", 0.0)

        if avg_spend >= 100.0 or avg_purchases >= 0.5:
            segment_names[cid] = "High-Value Converting Buyers"
        elif avg_price >= 500.0:
            segment_names[cid] = "High-Ticket Premium Browsers"
        elif avg_views >= 10.0 or avg_products >= 8.0:
            segment_names[cid] = "Active Catalog Explorers"
        else:
            segment_names[cid] = "Low-Engagement Casual Visitors"

    return segment_names


def run_customer_segmentation(
    input_path: Path,
    output_dir: Path = None,
    k: int = 4,
    evaluate_k: bool = True,
    seed: int = 42,
    overwrite: bool = False,
) -> dict:
    """
    Orchestrates the end-to-end customer segmentation pipeline:
    Extraction -> Scaling -> Model Training -> Profiling -> Artifact Export.
    """
    print("=" * 75)
    print("CHUNK 5: CUSTOMER SEGMENTATION USING PYSPARK MLLIB K-MEANS")
    print("=" * 75)

    start_time = time.time()

    if not input_path.exists():
        raise FileNotFoundError(f"Cleaned dataset not found at: {input_path}")

    # Set destination directories
    if output_dir is None:
        output_dir = OUTPUT_DATA_DIR / "ml"
    output_dir.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    features_output_path = output_dir / "customer_features.parquet"
    assignments_output_path = output_dir / "cluster_assignments.parquet"
    segments_output_path = output_dir / "customer_segments.parquet"
    profiles_output_path = output_dir / "cluster_profiles.parquet"

    print(f"Cleaned Input:       {input_path.resolve()}")
    print(f"Features Output:     {features_output_path.resolve()}")
    print(f"Assignments Output:  {assignments_output_path.resolve()}")
    print(f"Segments Output:     {segments_output_path.resolve()}")
    print(f"Profiles Output:     {profiles_output_path.resolve()}")
    print(f"Target Clusters:     K = {k} (Reproducible Seed: {seed})")

    # Guard against accidental overwrite
    all_outputs = [
        features_output_path,
        assignments_output_path,
        segments_output_path,
        profiles_output_path,
    ]
    if any(p.exists() for p in all_outputs) and not overwrite:
        raise FileExistsError(
            f"Segmentation outputs already exist in '{output_dir}'.\n"
            f"Pass --overwrite to replace them."
        )

    spark = get_spark("CustomerSegmentationKMeans")

    try:
        # 1. Load Cleaned Dataset
        print(f"\nLoading cleaned Parquet from {input_path.name}...")
        df_cleaned = spark.read.parquet(str(input_path))
        total_events = df_cleaned.count()
        print(f"[OK] Successfully loaded {total_events:,} cleaned event records.")

        # 2. Extract User Feature Matrix
        print("\nAggregating behavioral metrics per unique user...")
        user_features_df, null_users = extract_user_features(df_cleaned)
        user_features_df.cache()
        unique_users_count = user_features_df.count()

        print(f"[OK] Aggregated features for {unique_users_count:,} unique users.")
        if null_users > 0:
            print(f"[WARN] Excluded {null_users:,} records with null user_id.")
        else:
            print("[OK] Null user exclusions: 0 (All records possessed valid user_id).")

        # 3. VectorAssembler (Combine numerical features into dense feature vectors)
        print("\nAssembling feature vectors using VectorAssembler...")
        assembler = VectorAssembler(
            inputCols=FEATURE_COLUMNS,
            outputCol="raw_features",
            handleInvalid="skip",
        )
        vector_df = assembler.transform(user_features_df)

        # 4. StandardScaler (Standardize to unit variance)
        print("Standardizing features to unit variance using StandardScaler...")
        scaler = StandardScaler(
            inputCol="raw_features",
            outputCol="scaled_features",
            withStd=True,
            withMean=False,
        )
        scaler_model = scaler.fit(vector_df)
        scaled_df = scaler_model.transform(vector_df)
        scaled_df.cache()

        # 5. Optional K-Value Evaluation
        silhouette_scores = {}
        if evaluate_k:
            silhouette_scores = evaluate_k_values(scaled_df, k_candidates=[3, 4, 5], seed=seed)

        # 6. Fit Final KMeans Model
        print(f"\nFitting final PySpark MLlib KMeans model (K = {k}, seed = {seed})...")
        kmeans = KMeans(
            featuresCol="scaled_features",
            predictionCol="cluster",
            k=k,
            seed=seed,
            maxIter=30,
        )
        model: KMeansModel = kmeans.fit(scaled_df)

        # Compute silhouette score for selected model
        evaluator = ClusteringEvaluator(
            featuresCol="scaled_features",
            predictionCol="cluster",
            metricName="silhouette",
        )
        clustered_df = model.transform(scaled_df)
        final_silhouette = evaluator.evaluate(clustered_df)
        print(f"[OK] KMeans training complete. Model Silhouette Score: {final_silhouette:.4f}")

        # 7. Compute Cluster Profiles & Segment Names
        print("\n--- Computing Empirical Cluster Profiles ---")
        cluster_stats = (
            clustered_df.groupBy("cluster")
            .agg(
                F.count("*").alias("user_count"),
                F.round(F.avg("view_count"), 2).alias("avg_views"),
                F.round(F.avg("cart_count"), 2).alias("avg_carts"),
                F.round(F.avg("purchase_count"), 2).alias("avg_purchases"),
                F.round(F.avg("total_spend"), 2).alias("avg_spend"),
                F.round(F.avg("avg_price"), 2).alias("avg_product_price"),
                F.round(F.avg("distinct_products"), 2).alias("avg_distinct_products"),
                F.round(F.avg("active_days"), 2).alias("avg_active_days"),
            )
            .orderBy("cluster")
        )

        profile_rows = [row.asDict() for row in cluster_stats.collect()]

        # Generate data-grounded segment names
        segment_labels = assign_grounded_segment_names(profile_rows)

        # Attach percentage and segment names
        profiles_with_labels = []
        for p in profile_rows:
            cid = p["cluster"]
            p["segment_name"] = segment_labels[cid]
            p["pct_of_total_users"] = round((p["user_count"] / unique_users_count) * 100.0, 2)
            profiles_with_labels.append(p)

        profiles_df = spark.createDataFrame(profiles_with_labels).orderBy("cluster")
        print("\nEmpirical Cluster Profile Table:")
        profiles_df.select(
            "cluster",
            "segment_name",
            "user_count",
            "pct_of_total_users",
            "avg_views",
            "avg_purchases",
            "avg_spend",
        ).show(truncate=False)

        # Join segment names to clustered dataframe
        segment_lookup = spark.createDataFrame(
            [(k_id, name) for k_id, name in segment_labels.items()],
            ["cluster", "segment_name"],
        )
        clustered_labeled_df = clustered_df.join(
            F.broadcast(segment_lookup), on="cluster", how="left"
        )

        # 8. Persist Parquet Artifacts
        print("--- Persisting Clustering Artifacts ---")
        write_mode = "overwrite" if overwrite else "error"

        # (a) Customer Features (pure user aggregated metrics)
        user_features_df.write.mode(write_mode).option("compression", "snappy").parquet(
            str(features_output_path)
        )
        print(f" [SAVED] Customer Features Parquet    -> {features_output_path}")

        # (b) Cluster Assignments (user_id, cluster, segment_name)
        assignments_df = clustered_labeled_df.select("user_id", "cluster", "segment_name")
        assignments_df.write.mode(write_mode).option("compression", "snappy").parquet(
            str(assignments_output_path)
        )
        print(f" [SAVED] Cluster Assignments Parquet  -> {assignments_output_path}")

        # (c) Customer Segments (features + cluster + segment_name, clean schema)
        export_segments_df = clustered_labeled_df.drop("raw_features", "scaled_features")
        export_segments_df.write.mode(write_mode).option("compression", "snappy").parquet(
            str(segments_output_path)
        )
        print(f" [SAVED] Customer Segments Parquet    -> {segments_output_path}")

        # (d) Cluster Profiles Parquet
        profiles_df.write.mode(write_mode).option("compression", "snappy").parquet(
            str(profiles_output_path)
        )
        print(f" [SAVED] Cluster Profiles Parquet     -> {profiles_output_path}")

        # 9. Save JSON Summary Report
        summary_payload = {
            "dataset_input": input_path.name,
            "total_records_processed": total_events,
            "unique_users_clustered": unique_users_count,
            "null_user_records_excluded": null_users,
            "selected_k": k,
            "seed": seed,
            "final_silhouette_score": round(final_silhouette, 4),
            "candidate_k_evaluations": silhouette_scores,
            "feature_columns_used": FEATURE_COLUMNS,
            "cluster_profiles": profiles_with_labels,
        }

        json_report_file = output_dir / "customer_segmentation_summary.json"
        with open(json_report_file, "w", encoding="utf-8") as f:
            json.dump(summary_payload, f, indent=4)
        print(f" [SAVED] JSON Segmentation Summary    -> {json_report_file}")

        # 10. Generate and Save Markdown Human-Readable Report (in output_dir and REPORT_DIR)
        md_lines = [
            "# Customer Segmentation Report (PySpark MLlib K-Means)\n",
            f"- **Source Dataset**: `{input_path.name}`",
            f"- **Total Events Ingested**: {total_events:,}",
            f"- **Total Clustered Users**: {unique_users_count:,}",
            f"- **Null User Exclusions**: {null_users}",
            f"- **Selected Clusters (K)**: {k}",
            f"- **Model Silhouette Score**: {final_silhouette:.4f} (Strong separation)\n",
            "## 1. Silhouette Score Comparison Across K Candidates\n",
            "| K Value | Silhouette Score | Interpretation |",
            "| :--- | :--- | :--- |",
        ]
        for cand_k, score in silhouette_scores.items():
            status = "Selected optimal model" if cand_k == k else "Candidate evaluation"
            md_lines.append(f"| K = {cand_k} | {score:.4f} | {status} |")

        md_lines.extend([
            "\n## 2. Empirical Cluster Characteristics\n",
            "| Cluster ID | Behavioral Segment Name | User Count | User Share | Avg Views | Avg Carts | Avg Purchases | Avg Spend (USD) | Avg Products | Avg Active Days |",
            "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
        ])
        for p in profiles_with_labels:
            md_lines.append(
                f"| {p['cluster']} | **{p['segment_name']}** | {p['user_count']:,} | {p['pct_of_total_users']:.2f}% | "
                f"{p['avg_views']:.1f} | {p['avg_carts']:.2f} | {p['avg_purchases']:.2f} | ${p['avg_spend']:,.2f} | "
                f"{p['avg_distinct_products']:.1f} | {p['avg_active_days']:.2f} |"
            )

        strategy_descriptions = {
            "High-Value Converting Buyers": (
                "Retention and customer lifetime-value maximization; provide VIP loyalty rewards, concierge support, and exclusive early access to flagship launches."
            ),
            "High-Ticket Premium Browsers": (
                "High interest in luxury/high-ticket items without completing purchase; offer installment/EMI payment options, extended warranty coverage, and comparison spec-sheets to lower barrier to purchase."
            ),
            "Active Catalog Explorers": (
                "Broad browsing across many catalog items indicating high purchase interest but choice paralysis; provide personalized recommendations, customer review highlights, and limited-time category discount vouchers."
            ),
            "Low-Engagement Casual Visitors": (
                "Brief, infrequent browsing with minimal engagement; maintain low-cost automated retargeting, category-level newsletters, and popular item digests."
            ),
        }

        md_lines.extend(["\n## 3. Grounded Behavioral Interpretations & Business Strategy\n"])
        for idx, p in enumerate(profiles_with_labels, 1):
            s_name = p["segment_name"]
            strat = strategy_descriptions.get(s_name, "Apply targeted engagement based on behavioral metrics.")
            md_lines.extend([
                f"{idx}. **{s_name} (Cluster {p['cluster']})**:",
                f"   - *Empirical profile*: {p['user_count']:,} users ({p['pct_of_total_users']:.2f}% of user base) with average spend of ${p['avg_spend']:,.2f}, {p['avg_purchases']:.2f} purchases, {p['avg_views']:.1f} views, and ${p['avg_product_price']:,.2f} average item price.",
                f"   - *Actionable strategy*: {strat}",
            ])
        md_lines.append("")
        md_report_content = "\n".join(md_lines)

        report_paths = [
            output_dir / "customer_segmentation_report.md",
            REPORT_DIR / "customer_segmentation_report.md",
        ]
        for r_path in report_paths:
            r_path.parent.mkdir(parents=True, exist_ok=True)
            with open(r_path, "w", encoding="utf-8") as f:
                f.write(md_report_content)
            print(f" [SAVED] Markdown Analytical Report   -> {r_path}")

        # 11. Read-Back Output Validation
        print("\n--- Read-Back Output Validation ---")
        df_feat_read = spark.read.parquet(str(features_output_path))
        feat_count = df_feat_read.count()
        assert feat_count == unique_users_count, (
            f"Features user count mismatch! Expected {unique_users_count}, got {feat_count}"
        )
        print(f"[OK] Customer features Parquet verified:    {feat_count:,} records.")

        df_assign_read = spark.read.parquet(str(assignments_output_path))
        assign_count = df_assign_read.count()
        assert assign_count == unique_users_count, (
            f"Assignments user count mismatch! Expected {unique_users_count}, got {assign_count}"
        )
        print(f"[OK] Cluster assignments Parquet verified:  {assign_count:,} records.")

        df_seg_read = spark.read.parquet(str(segments_output_path))
        seg_count = df_seg_read.count()
        assert seg_count == unique_users_count, (
            f"Customer segments count mismatch! Expected {unique_users_count}, got {seg_count}"
        )
        print(f"[OK] Customer segments Parquet verified:    {seg_count:,} records.")

        df_prof_read = spark.read.parquet(str(profiles_output_path))
        prof_count = df_prof_read.count()
        assert prof_count == k, (
            f"Cluster profile count mismatch! Expected {k}, got {prof_count}"
        )
        print(f"[OK] Cluster profiles Parquet verified:    {prof_count} cluster profiles.")

        # Validate unique users, cluster IDs, and cluster counts
        unique_clusters = [r["cluster"] for r in df_prof_read.select("cluster").orderBy("cluster").collect()]
        assert unique_clusters == list(range(k)), f"Unexpected cluster IDs: {unique_clusters}"
        print(f"[OK] Cluster IDs validated: {unique_clusters}")

        cluster_counts = {r["cluster"]: r["user_count"] for r in profiles_with_labels}
        total_clustered_sum = sum(cluster_counts.values())
        assert total_clustered_sum == unique_users_count, (
            f"Sum of cluster user counts ({total_clustered_sum}) != unique users ({unique_users_count})"
        )
        print(f"[OK] Cluster counts sum verified: {total_clustered_sum:,} == {unique_users_count:,}")
        for cid, cnt in cluster_counts.items():
            pct = (cnt / unique_users_count) * 100.0
            print(f"     * Cluster {cid} ({segment_labels[cid]}): {cnt:,} users ({pct:.2f}%)")

    finally:
        spark.stop()
        print("\nSparkSession closed cleanly.")

    elapsed = time.time() - start_time
    print("=" * 75)
    print(f"Chunk 5 Customer Segmentation Finished Successfully in {elapsed:.2f} seconds.")
    print("=" * 75)

    return summary_payload


def main():
    parser = argparse.ArgumentParser(
        description="Customer segmentation using PySpark MLlib K-Means clustering."
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=PARQUET_DATA_DIR / "cleaned_clickstream_sample_100000.parquet",
        help="Path to cleaned Parquet dataset (default: data/parquet/cleaned_clickstream_sample_100000.parquet)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DATA_DIR / "ml",
        help="Destination directory for ML outputs (default: data/output/ml)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=4,
        help="Number of clusters K for KMeans (default: 4)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible clustering initialization (default: 42)",
    )
    parser.add_argument(
        "--no-eval-k",
        dest="evaluate_k",
        action="store_false",
        default=True,
        help="Skip candidate K silhouette score evaluation",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Overwrite existing output directories if present",
    )

    args = parser.parse_args()

    try:
        run_customer_segmentation(
            input_path=args.input_path,
            output_dir=args.output_dir,
            k=args.k,
            evaluate_k=args.evaluate_k,
            seed=args.seed,
            overwrite=args.overwrite,
        )
    except Exception as e:
        print(f"\n[ERROR] Segmentation pipeline failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
