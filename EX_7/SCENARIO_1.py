print("NIVASRI T | 24BAD081")
# 1. Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# 2. Load dataset
data = pd.read_csv(r"C:\Users\nivas\Downloads\archive (11)\Mall_Customers.csv")

print(data.head())

# 3. Data preprocessing
print(data.isnull().sum())

# select useful features
X = data[['Annual Income (k$)', 'Spending Score (1-100)']]

# scale data (important for K-Means)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Elbow Method to find best K
inertia = []

for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)=
# plot elbow curve
plt.plot(range(1, 11), inertia, marker='o')
plt.title("Elbow Method")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.show()

# 5. Apply K-Means (choose K = 5 from elbow)
kmeans = KMeans(n_clusters=5, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# 6. Assign cluster labels
data['Cluster'] = clusters

# 7. Calculate Silhouette Score
score = silhouette_score(X_scaled, clusters)
print("Silhouette Score:", score)

# 8. Visualize clusters
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters)
plt.title("Customer Clusters")
plt.xlabel("Annual Income (scaled)")
plt.ylabel("Spending Score (scaled)")

# plot centroids
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1], marker='X', s=200)

plt.show()

# 9. Interpret clusters (mean values)
print("\nCluster Analysis:")
print(data.groupby('Cluster')[['Annual Income (k$)', 'Spending Score (1-100)']].mean())