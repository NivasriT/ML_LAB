print("NIVASRI T | 24BAD081")
import pandas as pd
import matplotlib.pyplot as plt
plt.ion()   # enable interactive plotting
# load dataset (tab separated)
data = pd.read_csv(
    r"C:/Users/nivas/Desktop/NIVA/SEM 4/ML/LAB/PY FILES/marketing_campaign.csv",
    sep="\t"
)
# view basic data
print(data.head())
print(data.isnull().sum())

# age calculation and bar plot
if "Year_Birth" in data.columns:
    data["Age"] = 2024 - data["Year_Birth"]
    plt.figure()
    plt.bar(
        data["Age"].value_counts().sort_index().index,
        data["Age"].value_counts().sort_index().values
    )
    plt.title("Age Distribution")
    plt.xlabel("Age")
    plt.ylabel("Count")
    plt.show()
# income box plot
if "Income" in data.columns:
    plt.figure()
    plt.boxplot(data["Income"].dropna())
    plt.title("Income Distribution")
    plt.ylabel("Income")
    plt.show()
# spending pattern bar plot
spending_cols = [
    "MntWines", "MntFruits", "MntMeatProducts",
    "MntFishProducts", "MntSweetProducts", "MntGoldProds"
]
available_cols = [c for c in spending_cols if c in data.columns]
if len(available_cols) > 0:
    plt.figure()
    plt.bar(available_cols, data[available_cols].mean())
    plt.title("Average Spending Pattern")
    plt.ylabel("Amount Spent")
    plt.show()
