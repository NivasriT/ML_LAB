print("NIVASRI T | 24BAD081")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
data = pd.read_csv(
    r"C:\Users\nivas\Desktop\NIVA\SEM 4\ML\LAB\PY FILES\ex3\auto-mpg.csv"
)

# Clean data
data.replace("?", np.nan, inplace=True)
data["horsepower"] = pd.to_numeric(data["horsepower"])
data.dropna(inplace=True)

# Feature and target
X = data[["horsepower"]]
y = data["mpg"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale feature
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

degrees = [2, 3, 4]
train_err = []
test_err = []

plt.figure()

for d in degrees:
    poly = PolynomialFeatures(degree=d)
    X_train_p = poly.fit_transform(X_train_s)
    X_test_p = poly.transform(X_test_s)

    model = LinearRegression()
    model.fit(X_train_p, y_train)

    y_train_pred = model.predict(X_train_p)
    y_test_pred = model.predict(X_test_p)

    train_err.append(mean_squared_error(y_train, y_train_pred))
    test_err.append(mean_squared_error(y_test, y_test_pred))

    X_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    X_range_s = scaler.transform(X_range)
    X_range_p = poly.transform(X_range_s)
    y_range_pred = model.predict(X_range_p)

    plt.plot(X_range, y_range_pred, label=f"Degree {d}")

    print(f"\nDegree {d}")
    print("MSE :", mean_squared_error(y_test, y_test_pred))
    print("RMSE:", np.sqrt(mean_squared_error(y_test, y_test_pred)))
    print("R2  :", r2_score(y_test, y_test_pred))

# Polynomial curves
plt.scatter(X, y, s=10, color="black")
plt.xlabel("Horsepower")
plt.ylabel("MPG")
plt.title("Polynomial Regression")
plt.legend()
plt.show()

# Train vs Test error
plt.figure()
plt.plot(degrees, train_err, marker="o", label="Train Error")
plt.plot(degrees, test_err, marker="o", label="Test Error")
plt.xlabel("Degree")
plt.ylabel("MSE")
plt.title("Overfitting vs Underfitting")
plt.legend()
plt.show()

# Ridge regression
poly = PolynomialFeatures(degree=4)
X_train_p = poly.fit_transform(X_train_s)
X_test_p = poly.transform(X_test_s)

ridge = Ridge(alpha=1.0)
ridge.fit(X_train_p, y_train)

y_ridge = ridge.predict(X_test_p)

print("\nRidge Regression (Degree 4)")
print("MSE :", mean_squared_error(y_test, y_ridge))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_ridge)))
print("R2  :", r2_score(y_test, y_ridge))
