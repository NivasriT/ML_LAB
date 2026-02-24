print("NIVASRI T  | 24BAD081")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Load dataset
data = pd.read_csv(r"C:\Users\nivas\Desktop\NIVA\SEM 4\ML\LAB\PY FILES\ex5\train_u6lujuX_CVtuZ9i (1).csv")

# Select required features + target
features = ['ApplicantIncome','LoanAmount','Credit_History','Education','Property_Area']
X = data[features]
y = data['Loan_Status']

# Handle missing values
X['LoanAmount'].fillna(X['LoanAmount'].mean(), inplace=True)
X['Credit_History'].fillna(X['Credit_History'].mode()[0], inplace=True)
X['Education'].fillna(X['Education'].mode()[0], inplace=True)
X['Property_Area'].fillna(X['Property_Area'].mode()[0], inplace=True)
y.fillna(y.mode()[0], inplace=True)

# Encode categorical variables
le = LabelEncoder()
X['Education'] = le.fit_transform(X['Education'])
X['Property_Area'] = le.fit_transform(X['Property_Area'])
y = le.fit_transform(y)  # Approved=1, Rejected=0

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train shallow tree
tree_shallow = DecisionTreeClassifier(max_depth=3, random_state=42)
tree_shallow.fit(X_train, y_train)

# Train deep tree
tree_deep = DecisionTreeClassifier(random_state=42)
tree_deep.fit(X_train, y_train)

# Predictions
y_pred_shallow = tree_shallow.predict(X_test)
y_pred_deep = tree_deep.predict(X_test)

# Evaluation (Shallow Tree)
print("Shallow Tree Results")
print("Accuracy:", accuracy_score(y_test, y_pred_shallow))
print("Precision:", precision_score(y_test, y_pred_shallow))
print("Recall:", recall_score(y_test, y_pred_shallow))
print("F1 Score:", f1_score(y_test, y_pred_shallow))

# Evaluation (Deep Tree)
print("\nDeep Tree Results")
print("Accuracy:", accuracy_score(y_test, y_pred_deep))
print("Precision:", precision_score(y_test, y_pred_deep))
print("Recall:", recall_score(y_test, y_pred_deep))
print("F1 Score:", f1_score(y_test, y_pred_deep))

# Confusion Matrix (Deep Tree)
cm = confusion_matrix(y_test, y_pred_deep)
sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Feature Importance
importance = tree_deep.feature_importances_
plt.bar(features, importance)
plt.xticks(rotation=45)
plt.title("Feature Importance")
plt.show()

# Tree Structure Plot (Shallow for clarity)
plt.figure(figsize=(10,6))
plot_tree(tree_shallow, feature_names=features, class_names=['Rejected','Approved'], filled=True)
plt.show()

# Overfitting Check
print("\nTraining Accuracy (Deep Tree):", tree_deep.score(X_train, y_train))
print("Testing Accuracy (Deep Tree):", tree_deep.score(X_test, y_test))