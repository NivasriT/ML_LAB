import pandas as pd
import matplotlib.pyplot as plt

plt.ion()

data = pd.read_csv(
    r"C:/Users/nivas/Desktop/NIVA/SEM 4/ML/LAB/PY FILES/marketing_campaign.csv",
    sep=";"
)

data.columns = data.columns.str.strip()

print(data.head())
print(data.isnull().sum())

if "Year_Birth" in data.columns:
    data["Age"] = 2024 - data["Year_Birth"]
    plt.figure()
    plt.bar(data["Age"].value_counts().sort_index().index,
            data["Age"].value_counts().sort_index().values)
    plt.title("Age Distribution")
    plt.xlabel("Age")
    plt.ylabel("Count")
    plt.show()
    plt.pause(0.1)

if "Income" in data.columns:
    plt.figure()
    plt.boxplot(data["Income"].dropna())
    plt.title("Income Distribution")
    plt.ylabel("Income")
    plt.show()
    plt.pause(0.1)

spending_cols = [
    "MntWines", "MntFruits", "MntMeatProducts",
    "MntFishProducts", "MntSweetProducts", "MntGoldProds"
]

available_cols = []
for col in spending_cols:
    if col in data.columns:
        available_cols.append(col)

if len(available_cols) > 0:
    plt.figure()
    plt.bar(available_cols, data[available_cols].mean())
    plt.title("Average Spending Pattern")
    plt.ylabel("Amount")
    plt.show()
    plt.pause(0.1)
