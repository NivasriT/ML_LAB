print("NIVASRI T | 24BAD081")
# Import libraries
import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv(
    r"C:\Users\nivas\Desktop\NIVA\SEM 4\ML\LAB\PY FILES\data.csv",
    encoding="ISO-8859-1"
)

# View first rows
print("First 5 rows")
print(data.head())

# View last rows
print("\nLast 5 rows")
print(data.tail())

# Dataset info
print("\nDataset Info")
print(data.info())

# Statistical summary
print("\nDescription")
print(data.describe())

# Check missing values
print("\nMissing Values")
print(data.isnull().sum())

# Sales per product (top 10)
sales = data.groupby("Description")["Quantity"].sum().head(10)

# Bar chart
plt.figure()
plt.bar(sales.index, sales.values)
plt.title("Sales per Product")
plt.xlabel("Product")
plt.ylabel("Quantity Sold")
plt.xticks(rotation=90)
plt.show()

# Line chart
plt.figure()
plt.plot(sales.index, sales.values)
plt.title("Sales Trend")
plt.xlabel("Product")
plt.ylabel("Quantity Sold")
plt.xticks(rotation=90)
plt.show()
