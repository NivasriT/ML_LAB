print("NIVASRI T  | 24BAD081")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from matplotlib.colors import ListedColormap

# Load dataset
data = pd.read_csv(r"C:\Users\nivas\Desktop\NIVA\SEM 4\ML\LAB\PY FILES\ex5\breast-cancer.csv")

# Select features
X = data[['radius_mean','texture_mean','perimeter_mean','area_mean','smoothness_mean']]
y = data['diagnosis']

# Encode labels (M=1, B=0)
le = LabelEncoder()
y = le.fit_transform(y)

# Scale features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Try different K values
accuracy_list = []
k_values = range(1, 16)

for k in k_values:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy_list.append(accuracy_score(y_test, y_pred))

# Plot Accuracy vs K
plt.plot(k_values, accuracy_list)
plt.xlabel("K Value")
plt.ylabel("Accuracy")
plt.title("Accuracy vs K")
plt.show()

# Choose best K
best_k = k_values[np.argmax(accuracy_list)]
print("Best K:", best_k)

# Train final model
model = KNeighborsClassifier(n_neighbors=best_k)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Misclassified cases
print("Misclassified cases:", np.sum(y_test != y_pred))

# Decision Boundary (first 2 features only)
X2 = X[:, :2]
X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y, test_size=0.2, random_state=42)

model2 = KNeighborsClassifier(n_neighbors=best_k)
model2.fit(X_train2, y_train2)

x_min, x_max = X2[:, 0].min()-1, X2[:, 0].max()+1
y_min, y_max = X2[:, 1].min()-1, X2[:, 1].max()+1

xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                     np.linspace(y_min, y_max, 200))

Z = model2.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.3, cmap=ListedColormap(['blue','red']))
plt.scatter(X2[:,0], X2[:,1], c=y, cmap=ListedColormap(['blue','red']))
plt.xlabel("radius_mean")
plt.ylabel("texture_mean")
plt.title("Decision Boundary")
plt.show()