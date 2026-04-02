print("NIVASRI T |24BAD081")
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns

# load dataset
df = pd.read_csv(r"C:\Users\nivas\Desktop\NIVA\SEM 4\ML\LAB\PY FILES\EX 6\diabetes_bagging.csv")

# split input and output
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# decision tree model
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)

# bagging model
bag = BaggingClassifier(estimator=DecisionTreeClassifier(), n_estimators=10, random_state=1)
bag.fit(X_train, y_train)
y_pred_bag = bag.predict(X_test)

# accuracy
acc_dt = accuracy_score(y_test, y_pred_dt)
acc_bag = accuracy_score(y_test, y_pred_bag)

print("Decision Tree Accuracy:", acc_dt)
print("Bagging Accuracy:", acc_bag)

# bar graph
models = ["Decision Tree", "Bagging"]
accuracy = [acc_dt, acc_bag]

plt.bar(models, accuracy)
plt.title("Accuracy Comparison")
plt.ylabel("Accuracy")
plt.show()

# confusion matrix for bagging
cm = confusion_matrix(y_test, y_pred_bag)

sns.heatmap(cm, annot=True, fmt='d')
plt.title("Confusion Matrix - Bagging")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()