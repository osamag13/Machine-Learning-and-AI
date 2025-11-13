# --- Import Required Libraries ---
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder

# --- Load Data ---
customers = pd.read_csv('testdata.csv')
print("First 5 rows of dataset:")
print(customers.head())

# --- Step 1: KMeans with Annual Income & Spending Score ---
points = customers.iloc[:, 3:5].values  
x = points[:, 0]
y = points[:, 1]

# Apply KMeans
kmeans = KMeans(n_clusters=5, random_state=0)
kmeans.fit(points)
predicted_cluster_indexes = kmeans.predict(points)

# --- Plot Clusters ---
plt.figure(figsize=(8, 6))
plt.scatter(x, y, c=predicted_cluster_indexes, s=50, alpha=0.7, cmap='viridis')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score')
plt.title('Customer Segmentation (Income vs Spending)')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=100, marker='X', label='Centroids')
plt.legend()
plt.grid(True)
plt.show()

# Save figure
plt.savefig('customersdataplot_kmeans.png')
print("KMeans cluster image saved as 'customersdataplot_kmeans.png'")

# --- Step 2: Predict Cluster for New Customer ---
new_customer = np.array([[120, 20]])
cluster = kmeans.predict(new_customer)[0]
clustered_df = customers.copy()
clustered_df['Cluster'] = kmeans.predict(points)

print(f"\nCustomer with (Annual Income = 120k$, Spending Score = 20) belongs to Cluster {cluster}")
print("Customers in the same cluster:")
print(clustered_df[clustered_df['Cluster'] == cluster]['CustomerID'].values)

# --- Step 3: Encode Gender ---
df = customers.copy()
encoder = LabelEncoder()
df['Gender'] = encoder.fit_transform(df['Gender'])

# --- Step 4: Use More Features for Clustering ---
points = df.iloc[:, 1:5].values

# --- Elbow Method to Find Optimal Clusters ---
inertias = []
for i in range(1, 10):
    kmeans = KMeans(n_clusters=i, random_state=0)
    kmeans.fit(points)
    inertias.append(kmeans.inertia_)

plt.figure(figsize=(8, 6))
plt.plot(range(1, 10), inertias, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
plt.grid(True)
plt.show()

plt.savefig('customersdataplot_elbow.png')
print("Elbow plot image saved as 'customersdataplot_elbow.png'")

# --- Step 5: Final KMeans with 5 Clusters ---
kmeans = KMeans(n_clusters=5, random_state=0)
kmeans.fit(points)
df['Cluster'] = kmeans.predict(points)

# --- Step 6: Compute Cluster Statistics ---
results = pd.DataFrame(columns=[
    'Cluster', 'Average Age', 'Average Income', 'Average Spending Index',
    'Number of Females', 'Number of Males'
])

for i, center in enumerate(kmeans.cluster_centers_):
    age = center[1]       # Average age
    income = center[2]    # Average income
    spend = center[3]     # Average spending score
    gdf = df[df['Cluster'] == i]
    females = gdf[gdf['Gender'] == 0].shape[0]
    males = gdf[gdf['Gender'] == 1].shape[0]
    results.loc[i] = [i, age, income, spend, females, males]

print("\n--- Cluster Summary ---")
print(results)

# --- Optional: Visualize Final Clusters (using two main features) ---
plt.figure(figsize=(8, 6))
plt.scatter(df['Annual Income (k$)'], df['Spending Score (1-100)'], c=df['Cluster'], cmap='viridis', s=50)
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title('Final KMeans Clusters')
plt.grid(True)
plt.show()

plt.savefig('final_clusters_plot.png')
print("Final clusters image saved as 'final_clusters_plot.png'")
