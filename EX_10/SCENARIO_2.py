print("NIVASRI T | 24BAD081")
# 1. Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF
from sklearn.metrics import mean_squared_error
from numpy.linalg import svd
import seaborn as sns

# 2. Load dataset
data = pd.read_csv(r"C:\Users\nivas\Downloads\archive (9)\ratings.csv")
data = data[['userId', 'movieId', 'rating']]

# 3. Create User-Item Matrix
user_item = data.pivot(index='userId', columns='movieId', values='rating')

# 4. Fill missing values
user_item_filled = user_item.fillna(0)

# 🔷 SVD MODEL

user_mean = user_item_filled.mean(axis=1)
centered = user_item_filled.sub(user_mean, axis=0)

U, sigma, Vt = svd(centered, full_matrices=False)
sigma = np.diag(sigma)

k = 20
svd_recon = np.dot(np.dot(U[:, :k], sigma[:k, :k]), Vt[:k, :])
svd_recon = svd_recon + user_mean.values.reshape(-1, 1)

# 🔷 NMF MODEL

nmf = NMF(n_components=20, random_state=42, max_iter=200)
W = nmf.fit_transform(user_item_filled)
H = nmf.components_

nmf_recon = np.dot(W, H)

# 🔷 EVALUATION (RMSE)

true = user_item.values
rows, cols = np.where(~np.isnan(true))

svd_rmse = np.sqrt(mean_squared_error(true[rows, cols], svd_recon[rows, cols]))
nmf_rmse = np.sqrt(mean_squared_error(true[rows, cols], nmf_recon[rows, cols]))

print("SVD RMSE:", svd_rmse)
print("NMF RMSE:", nmf_rmse)

# 🔷 Precision@K & Recall@K

def precision_recall_at_k(pred, true, user_id, k=5, threshold=3.5):
    user_true = true[user_id - 1]
    user_pred = pred[user_id - 1]

    # actual relevant items
    relevant = set(np.where(user_true >= threshold)[0])

    # top-k predicted items
    top_k = np.argsort(user_pred)[-k:]

    recommended = set(top_k)

    # intersection
    hits = len(recommended & relevant)

    precision = hits / k
    recall = hits / len(relevant) if len(relevant) > 0 else 0

    return precision, recall

p, r = precision_recall_at_k(nmf_recon, true, user_id=1)

print("\nPrecision@5:", p)
print("Recall@5:", r)

# 🔷 VISUALIZATION

# 1. Reconstruction comparison
plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
sns.heatmap(user_item_filled.iloc[:20,:20])
plt.title("Original Matrix")

plt.subplot(1,2,2)
sns.heatmap(nmf_recon[:20,:20])
plt.title("NMF Reconstructed")

plt.show()

# 2. Latent feature visualization
plt.plot(W[:10])
plt.title("User Latent Features (NMF)")
plt.xlabel("Features")
plt.ylabel("Value")
plt.show()

# 3. Recommendation ranking chart
user_id = 1
user_pred = nmf_recon[user_id - 1]

top_items = np.argsort(user_pred)[-5:]
top_scores = user_pred[top_items]

plt.bar(range(5), top_scores)
plt.title("Top-5 Recommendations Scores")
plt.xlabel("Items")
plt.ylabel("Predicted Rating")
plt.show()