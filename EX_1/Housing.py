print("NIVASRI T | 24BAD081")
# Import libraries
import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv(
    r"C:/Users/nivas/Desktop/NIVA/SEM 4/ML/LAB/PY FILES/Housing.csv"
)
# Show columns
print("Columns:")
print(data.columns)

# Show first 5 rows
print("\nFirst 5 rows:")
print(data.head())

# Show last 5 rows
print("\nLast 5 rows:")
print(data.tail())

# Check missing values
print("\nMissing values:")
print(data.isnull().sum())

# Scatter plot: Area vs Price
plt.figure()
plt.scatter(data["area"], data["price"])
plt.title("Area vs Price")
plt.xlabel("Area")
plt.ylabel("Price")
plt.show()

# Correlation heatmap using only numeric columns
numeric_data = data.select_dtypes(include='number')
corr = numeric_data.corr()

plt.figure()
plt.imshow(corr)
plt.colorbar()
plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
plt.yticks(range(len(corr.columns)), corr.columns)
plt.title("Correlation Heatmap")
plt.show()
