print("NIVASRI T | 24BAD081")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score

# 1. Load Dataset
data = pd.read_csv(
    r"C:\Users\nivas\Desktop\NIVA\SEM 4\ML\LAB\PY FILES\ex3\StudentsPerformance.csv"
)
# 2. Encode Categorical Features

le = LabelEncoder()
data["parental level of education"] = le.fit_transform(
    data["parental level of education"]
)
data["test preparation course"] = le.fit_transform(
    data["test preparation course"]
)
# 3. Create Target Variable (Average Score)

data["Final_Score"] = (
    data["math score"] +
    data["reading score"] +
    data["writing score"]
) / 3
# 4. Select Input Features

X = data[
    [
        "math score",
        "reading score",
        "writing score",
        "parental level of education",
        "test preparation course"
    ]
]

y = data["Final_Score"]
# 5. Handle Missing Values

X = X.fillna(X.mean())
# 6. Feature Scaling

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# 7. Train-Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)
# 8. Train Multilinear Regression Model

model = LinearRegression()
model.fit(X_train, y_train)

# 9. Predictions

y_pred = model.predict(X_test)

# 10. Evaluation Metrics

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("MSE :", mse)
print("RMSE:", rmse)
print("R2  :", r2)

# 11. Regression Coefficients

coeff_df = pd.DataFrame(
    {
        "Feature": X.columns,
        "Coefficient": model.coef_
    }
)
print("\nRegression Coefficients:")
print(coeff_df)

# 12. Ridge Regression

ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)

# 13. Lasso Regression

lasso = Lasso(alpha=0.01)
lasso.fit(X_train, y_train)

# ---------------- VISUALIZATIONS -----------------
# Predicted vs Actual
plt.figure()
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Score")
plt.ylabel("Predicted Score")
plt.title("Actual vs Predicted Exam Scores")
plt.show()

# Coefficient Magnitude Comparison
plt.figure()
plt.bar(X.columns, model.coef_)
plt.xticks(rotation=45)
plt.xlabel("Features")
plt.ylabel("Coefficient Value")
plt.title("Feature Coefficient Magnitudes")
plt.show()

# Residual Distribution
residuals = y_test - y_pred

plt.figure()
plt.hist(residuals, bins=20)
plt.xlabel("Residual Error")
plt.ylabel("Frequency")
plt.title("Residual Distribution")
plt.show()
