print("NIVASRI T | 24BAD081")
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, precision_recall_curve
from imblearn.over_sampling import SMOTE

# load dataset
df = pd.read_csv("fraud_smote.csv")   # change path if needed

# split data
X = df.drop("Fraud", axis=1)
y = df["Fraud"]

# check class distribution
print("Before SMOTE:\n", y.value_counts())

# plot before SMOTE
y.value_counts().plot(kind='bar')
plt.title("Before SMOTE")
plt.show()

# train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# model before SMOTE
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Before SMOTE Report:\n", classification_report(y_test, y_pred))

# apply SMOTE
smote = SMOTE(random_state=1)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

# check after SMOTE
print("After SMOTE:\n", pd.Series(y_train_sm).value_counts())

# plot after SMOTE
pd.Series(y_train_sm).value_counts().plot(kind='bar')
plt.title("After SMOTE")
plt.show()

# model after SMOTE
model_sm = LogisticRegression(max_iter=1000)
model_sm.fit(X_train_sm, y_train_sm)
y_pred_sm = model_sm.predict(X_test)

print("After SMOTE Report:\n", classification_report(y_test, y_pred_sm))

# precision-recall curve
y_prob = model_sm.predict_proba(X_test)[:,1]
precision, recall, _ = precision_recall_curve(y_test, y_prob)

plt.plot(recall, precision)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.show()