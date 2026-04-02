("NIVASRI T | 24BAD081")
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import accuracy_score

# load dataset
df = pd.read_csv("heart_stacking.csv")   # change path if needed

# encode categorical columns (if any)
le = LabelEncoder()
for col in df.select_dtypes(include="object"):
    df[col] = le.fit_transform(df[col])

# split data
X = df.drop("HeartDisease", axis=1)   # change if column name differs
y = df["HeartDisease"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# base models
lr = LogisticRegression(max_iter=1000)
svm = SVC(probability=True)
dt = DecisionTreeClassifier()

# train base models
lr.fit(X_train, y_train)
svm.fit(X_train, y_train)
dt.fit(X_train, y_train)

# predictions
y_lr = lr.predict(X_test)
y_svm = svm.predict(X_test)
y_dt = dt.predict(X_test)

# accuracy
acc_lr = accuracy_score(y_test, y_lr)
acc_svm = accuracy_score(y_test, y_svm)
acc_dt = accuracy_score(y_test, y_dt)

print("Logistic Regression:", acc_lr)
print("SVM:", acc_svm)
print("Decision Tree:", acc_dt)

# stacking model
estimators = [
    ('lr', lr),
    ('svm', svm),
    ('dt', dt)
]

stack = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())
stack.fit(X_train, y_train)

y_stack = stack.predict(X_test)
acc_stack = accuracy_score(y_test, y_stack)

print("Stacking Accuracy:", acc_stack)

# bar chart
models = ["LR", "SVM", "DT", "Stacking"]
accuracy = [acc_lr, acc_svm, acc_dt, acc_stack]

plt.bar(models, accuracy)
plt.title("Model Comparison")
plt.ylabel("Accuracy")
plt.show()