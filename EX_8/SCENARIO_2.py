print("NIVASRI T | 24BAD081")
# 1. Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 2. Load dataset
data = pd.read_csv(r"C:\Users\nivas\Downloads\archive (10)\Wine dataset.csv")

print("Dataset Preview:")
print(data.head())

# 3. Preprocessing (handle missing values)
data = data.dropna()

# 4. Separate features (drop target if exists)
# assuming last column is label (if not, adjust)
X = data.iloc[:, :-1]

# 5. Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 6. Apply PCA
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# 7. Explained variance
explained_variance = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)

print("\nExplained Variance Ratio:")
print(explained_variance)

print("\nCumulative Variance:")
print(cumulative_variance)

# 8. Reduce to 2 components
pca_2 = PCA(n_components=2)
X_reduced = pca_2.fit_transform(X_scaled)

# VISUALIZATION

# 1. Scree Plot
plt.figure()
plt.plot(range(1, len(explained_variance)+1), explained_variance, marker='o')
plt.xlabel("Principal Components")
plt.ylabel("Variance Explained")
plt.title("Scree Plot")
plt.show()

# 2. Cumulative Variance Plot
plt.figure()
plt.plot(range(1, len(cumulative_variance)+1), cumulative_variance, marker='o')
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Variance")
plt.title("Cumulative Variance Plot")
plt.show()

# 3. 2D Scatter Plot
plt.figure()
plt.scatter(X_reduced[:, 0], X_reduced[:, 1])
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA 2D Visualization")
plt.show()