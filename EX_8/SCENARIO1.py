print("NIVASRI T | 24BAD081")
# 1. Import libraries
import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import networkx as nx

# 2. Load dataset
data = pd.read_csv(r"C:\Users\nivas\Downloads\archive (12)\Groceries_dataset.csv")

print("Dataset Preview:")
print(data.head())

# 3. Create transactions (group by customer)
transactions = data.groupby('Member_number')['itemDescription'] \
                   .apply(list).values.tolist()

# 4. One-hot encoding
te = TransactionEncoder()
te_data = te.fit(transactions).transform(transactions)

df = pd.DataFrame(te_data, columns=te.columns_)

# 5. Apply Apriori (set support)
frequent_itemsets = apriori(df, min_support=0.01, use_colnames=True)

print("\nFrequent Itemsets:")
print(frequent_itemsets.head())

# 6. Generate association rules
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.2)

# 7. Filter strong rules
rules = rules[rules['lift'] > 1]

print("\nStrong Rules:")
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head())

# VISUALIZATION

# 1. Bar chart of frequent itemsets
top_items = frequent_itemsets.sort_values(by='support', ascending=False).head(10)

plt.figure()
plt.bar(range(len(top_items)), top_items['support'])
plt.xticks(range(len(top_items)), top_items['itemsets'].astype(str), rotation=60)
plt.title("Top Frequent Itemsets")
plt.xlabel("Itemsets")
plt.ylabel("Support")
plt.show()

# 2. Support vs Confidence plot
if len(rules) > 0:
    plt.figure()
    plt.scatter(rules['support'], rules['confidence'])
    plt.xlabel("Support")
    plt.ylabel("Confidence")
    plt.title("Support vs Confidence")
    plt.show()
else:
    print("No rules to plot")

# 3. Network graph of association rules
if len(rules) > 0:
    G = nx.DiGraph()

    # add edges
    for _, row in rules.head(10).iterrows():
        for a in row['antecedents']:
            for c in row['consequents']:
                G.add_edge(a, c)

    plt.figure(figsize=(8,6))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True)
    plt.title("Association Rules Network")
    plt.show()
else:
    print("No rules for network graph")

# EVALUATION METRICS OUTPUT

print("\nEvaluation Metrics (Sample):")
print(rules[['support', 'confidence', 'lift']].head())