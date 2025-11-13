Customer Segmentation using K-Means Clustering

This project performs customer segmentation using K-Means clustering to identify distinct customer groups based on demographic and spending behavior data.

The code:

  Loads and preprocesses customer data (from testdata.csv),
  Applies K-Means clustering using: Annual Income (k$), Spending Score (1–100) and later additional features (Age, Gender, etc.),
  Visualizes clusters and centroids using Matplotlib,
  Determines the optimal number of clusters using the Elbow Method,
  Computes per-cluster statistics such as: Average Age, Average Income, Average Spending Score, Gender distribution (number of males and females)

Visualizations Generated

  Customer Segmentation Plot (customersdataplot_kmeans.png) {Shows the 5 clusters of customers based on Income and Spending Score.},
  Elbow Method Plot (customersdataplot_elbow.png) {Helps identify the optimal number of clusters by plotting inertia vs. number of clusters.}
  Final Cluster Visualization (final_clusters_plot.png) {Displays the final K-Means segmentation with 5 clusters.}

Technologies Used

  Python 3, 
  Pandas, NumPy — Data handling and preprocessing,
  Scikit-learn (KMeans, LabelEncoder) — Clustering and encoding, 
  Matplotlib — Visualization

Key Outputs

  Assigns each customer to a cluster,
  Predicts the cluster for a new customer (example: Income = 120k$, Spending Score = 20),
  Generates a summary table showing per-cluster insights: Average Age, Average Income, Average Spending Index,  Number of Males & Females
