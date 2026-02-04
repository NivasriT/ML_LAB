print("NIVASRI T | 24BAD081")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, roc_curve, auc

# File path (use ANY one file at a time)
path = r"C:\Users\nivas\Desktop\NIVA\SEM 4\ML\LAB\PY FILES\ex2\LICI - Daily data.csv"

# Load data
data = pd.read_csv(path)

# ---- STANDARDIZE COLUMN NAMES ----
data.columns = data.columns.str.strip().str.lower()

# Rename common variants
data.rename(columns={
    "open price": "open",
    "close price": "close",
    "volume shares": "volume"
}, inplace=True)

# ---- CREATE TARGET VARIABLE ----
data["price_movement"] = np.where(
    data["close"] > data["open"], 1, 0
)

# ---- SELECT FEATURES ----
X = data.select_dtypes(include=["int64", "float64"])
X = X.drop(columns=["close", "price_movement"], errors="ignore")
y = data["price_movement"]

# ---- HANDLE MISSING VALUES ----
X.fillna(X.mean(), inplace=True)

# ---- FEATURE SCALING ----
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---- TRAIN TEST SPLIT ----
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# ---- LOGISTIC REGRESSION ----
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# ---- PREDICTION ----
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# ---- EVALUATION ----
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# ---- ROC CURVE ----
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, label="AUC = %.2f" % roc_auc)
plt.plot([0, 1], [0, 1])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()

# ---- FEATURE IMPORTANCE ----
plt.bar(X.columns, model.coef_[0])
plt.xticks(rotation=45)
plt.show()

# ---- HYPERPARAMETER TUNING ----
param_grid = {
    "C": [0.01, 0.1, 1, 10],
    "penalty": ["l1", "l2"],
    "solver": ["liblinear"]
}

grid = GridSearchCV(LogisticRegression(max_iter=1000),
                    param_grid, cv=5)
grid.fit(X_train, y_train)

print("Best Parameters:", grid.best_params_)
