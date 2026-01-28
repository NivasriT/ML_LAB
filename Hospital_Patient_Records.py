print("NIVASRI T | 24BAD081")
# Import libraries
import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv(
    r"C:/Users/nivas/Desktop/NIVA/SEM 4/ML/LAB/PY FILES/diabetes.csv"
)

# Show first 5 rows
print("First 5 rows:")
print(data.head())

# Show last 5 rows
print("\nLast 5 rows:")
print(data.tail())

# Dataset structure
print("\nDataset Info:")
print(data.info())

# Check missing values
print("\nMissing values:")
print(data.isnull().sum())

# Histogram - Glucose
plt.figure()
plt.hist(data["Glucose"], bins=15)
plt.title("Glucose Distribution")
plt.xlabel("Glucose")
plt.ylabel("Count")
plt.show()

# Boxplot - Age
plt.figure()
plt.boxplot(data["Age"])
plt.title("Age Distribution")
plt.ylabel("Age")
plt.show()
