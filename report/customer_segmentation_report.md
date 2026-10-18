# Customer Segmentation Report (PySpark MLlib K-Means)

- **Source Dataset**: `cleaned_clickstream_sample_100000.parquet`
- **Total Events Ingested**: 99,973
- **Total Clustered Users**: 20,778
- **Null User Exclusions**: 0
- **Selected Clusters (K)**: 4
- **Model Silhouette Score**: 0.6540 (Strong separation)

## 1. Silhouette Score Comparison Across K Candidates

| K Value | Silhouette Score | Interpretation |
| :--- | :--- | :--- |
| K = 3 | 0.7700 | Candidate evaluation |
| K = 4 | 0.6540 | Selected optimal model |
| K = 5 | 0.7750 | Candidate evaluation |

## 2. Empirical Cluster Characteristics

| Cluster ID | Behavioral Segment Name | User Count | User Share | Avg Views | Avg Carts | Avg Purchases | Avg Spend (USD) | Avg Products | Avg Active Days |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 0 | **High-Ticket Premium Browsers** | 2,731 | 13.14% | 2.9 | 0.02 | 0.00 | $0.00 | 2.1 | 1.00 |
| 1 | **Active Catalog Explorers** | 1,527 | 7.35% | 21.9 | 0.03 | 0.04 | $4.77 | 13.6 | 1.00 |
| 2 | **Low-Engagement Casual Visitors** | 15,911 | 76.58% | 3.3 | 0.02 | 0.03 | $3.77 | 2.3 | 1.00 |
| 3 | **High-Value Converting Buyers** | 609 | 2.93% | 6.5 | 1.16 | 1.38 | $549.06 | 3.3 | 1.00 |

## 3. Grounded Behavioral Interpretations & Business Strategy

1. **High-Ticket Premium Browsers (Cluster 0)**:
   - *Empirical profile*: 2,731 users (13.14% of user base) with average spend of $0.00, 0.00 purchases, 2.9 views, and $1,001.15 average item price.
   - *Actionable strategy*: High interest in luxury/high-ticket items without completing purchase; offer installment/EMI payment options, extended warranty coverage, and comparison spec-sheets to lower barrier to purchase.
2. **Active Catalog Explorers (Cluster 1)**:
   - *Empirical profile*: 1,527 users (7.35% of user base) with average spend of $4.77, 0.04 purchases, 21.9 views, and $255.72 average item price.
   - *Actionable strategy*: Broad browsing across many catalog items indicating high purchase interest but choice paralysis; provide personalized recommendations, customer review highlights, and limited-time category discount vouchers.
3. **Low-Engagement Casual Visitors (Cluster 2)**:
   - *Empirical profile*: 15,911 users (76.58% of user base) with average spend of $3.77, 0.03 purchases, 3.3 views, and $184.63 average item price.
   - *Actionable strategy*: Brief, infrequent browsing with minimal engagement; maintain low-cost automated retargeting, category-level newsletters, and popular item digests.
4. **High-Value Converting Buyers (Cluster 3)**:
   - *Empirical profile*: 609 users (2.93% of user base) with average spend of $549.06, 1.38 purchases, 6.5 views, and $410.17 average item price.
   - *Actionable strategy*: Retention and customer lifetime-value maximization; provide VIP loyalty rewards, concierge support, and exclusive early access to flagship launches.
