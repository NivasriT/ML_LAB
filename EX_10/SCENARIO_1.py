print("NIVASRI T | 24BAD081")
# 1. Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from numpy.linalg import svd
import seaborn as sns

# 2. Load dataset (ratings.csv)
data = pd.read_csv(r"C:\Users\nivas\Downloads\archive (9)\ratings.csv")

# check columns
print(data.head())
print(data.columns)

# 3. Keep only needed columns
data = data[['userId', 'movieId', 'rating']]

# 4. Create User-Item Matrix
user_item = data.pivot(index='userId', columns='movieId', values='rating')

# 5. Fill missing values with 0 (for SVD)
user_item_filled = user_item.fillna(0)

# 6. Normalize (mean centering)
user_mean = user_item_filled.mean(axis=1)
user_item_centered = user_item_filled.sub(user_mean, axis=0)

# 7. Apply SVD
U, sigma, Vt = svd(user_item_centered, full_matrices=False)
sigma = np.diag(sigma)

# 8. Reduce dimensions (latent factors)
k = 20  # you can change this
U_k = U[:, :k]
sigma_k = sigma[:k, :k]
Vt_k = Vt[:k, :]

# 9. Reconstruct matrix
reconstructed = np.dot(np.dot(U_k, sigma_k), Vt_k)

# add mean back
reconstructed = reconstructed + user_mean.values.reshape(-1, 1)

predicted_ratings = pd.DataFrame(reconstructed,
                                 index=user_item.index,
                                 columns=user_item.columns)

# 10. Evaluate (CORRECT METHOD - NO ERROR)
true = user_item.values
pred = predicted_ratings.values

rows, cols = np.where(~np.isnan(true))

true_vals = true[rows, cols]
pred_vals = pred[rows, cols]

rmse = np.sqrt(mean_squared_error(true_vals, pred_vals))
mae = mean_absolute_error(true_vals, pred_vals)

print("\nRMSE:", rmse)
print("MAE:", mae)

# 11. Top-N Recommendations
user_id = 1  # change user

user_pred = predicted_ratings.loc[user_id]
already_rated = user_item.loc[user_id].dropna().index

recommendations = user_pred.drop(already_rated)

top_n = recommendations.sort_values(ascending=False).head(5)

print("\nTop 5 Recommendations for User", user_id)
print(top_n)

# 12. Visualization – Heatmaps
plt.figure(figsize=(8,5))
sns.heatmap(user_item_filled.iloc[:20, :20])
plt.title("Original Matrix")
plt.show()

plt.figure(figsize=(8,5))
sns.heatmap(predicted_ratings.iloc[:20, :20])
plt.title("Reconstructed Matrix")
plt.show()

# 13. Error vs Latent Factors (k)
ks = [5, 10, 20, 30]
errors = []

for k in ks:
    U_k = U[:, :k]
    sigma_k = sigma[:k, :k]
    Vt_k = Vt[:k, :]

    recon = np.dot(np.dot(U_k, sigma_k), Vt_k)
    recon = recon + user_mean.values.reshape(-1, 1)

    rows, cols = np.where(~np.isnan(true))
    rmse_k = np.sqrt(mean_squared_error(true[rows, cols], recon[rows, cols]))
    errors.append(rmse_k)

plt.plot(ks, errors, marker='o')
plt.xlabel("k (latent factors)")
plt.ylabel("RMSE")
plt.title("Error vs k")
plt.show()