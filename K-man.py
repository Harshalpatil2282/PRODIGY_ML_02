# 1. Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 2. Load dataset
# Example: Mall Customer Segmentation dataset
df = pd.read_csv("Mall_Customers.csv")

# 3. Select relevant features (e.g., Annual Income and Spending Score)
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# 4. Normalize data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5. Find optimal number of clusters using Elbow Method
inertia = []
K = range(1, 11)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8, 4))
plt.plot(K, inertia, 'bx-')
plt.xlabel('k')
plt.ylabel('Inertia')
plt.title('Elbow Method For Optimal k')
plt.show()

# 6. Apply KMeans with optimal k (let’s assume k=5 from elbow plot)
kmeans = KMeans(n_clusters=5, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# 7. Add cluster labels to the original dataset
df['Cluster'] = clusters

# 8. Visualize clusters
plt.figure(figsize=(8, 5))
for cluster in range(5):
    plt.scatter(
        X_scaled[clusters == cluster, 0],
        X_scaled[clusters == cluster, 1],
        label=f'Cluster {cluster}'
    )

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            s=200, c='black', label='Centroids', marker='X')
plt.title("Customer Segments")
plt.xlabel("Annual Income (scaled)")
plt.ylabel("Spending Score (scaled)")
plt.legend()
plt.grid(True)
plt.show()
