print("NIVASRI T | 24BAD081")
# 1. Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# 2. Load dataset
data = pd.read_csv(r"C:\Users\nivas\Downloads\archive (11)\Mall_Customers.csv")

# 3. Select features
X = data[['Annual Income (k$)', 'Spending Score (1-100)']]

# 4. Scale data (important)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5. Choose number of components (try 2–10)
aic = []
bic = []

for k in range(2, 11):
    gmm = GaussianMixture(n_components=k, random_state=42)
    gmm.fit(X_scaled)
    aic.append(gmm.aic(X_scaled))
    bic.append(gmm.bic(X_scaled))

# plot AIC & BIC
plt.plot(range(2, 11), aic, label='AIC')
plt.plot(range(2, 11), bic, label='BIC')
plt.xlabel("Number of Clusters")
plt.ylabel("Score")
plt.title("AIC vs BIC")
plt.legend()
plt.show()

# 6. Apply GMM (choose k=5)
gmm = GaussianMixture(n_components=5, random_state=42)
gmm.fit(X_scaled)

# 7. Predict probabilities (soft clustering)
probs = gmm.predict_proba(X_scaled)

# 8. Assign cluster labels (highest probability)
clusters = np.argmax(probs, axis=1)
data['GMM_Cluster'] = clusters

# 9. Evaluate
print("Log-Likelihood:", gmm.score(X_scaled))
print("Silhouette Score:", silhouette_score(X_scaled, clusters))

# 10. Visualize clusters
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters)
plt.title("GMM Clustering")
plt.xlabel("Annual Income (scaled)")
plt.ylabel("Spending Score (scaled)")
plt.show()

# 11. Probability distribution (soft clustering)
plt.hist(probs.max(axis=1), bins=10)
plt.title("Cluster Confidence (Probability)")
plt.xlabel("Max Probability")
plt.ylabel("Count")
plt.show()

# 12. Compare with K-Means
kmeans = KMeans(n_clusters=5, random_state=42)
k_clusters = kmeans.fit_predict(X_scaled)

plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=k_clusters)
plt.title("K-Means Clustering")
plt.show()