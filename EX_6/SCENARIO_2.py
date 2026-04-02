print("NIVASRI | 24BAD081") 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_curve, auc

# load dataset
df = pd.read_csv(r"C:\Users\nivas\Desktop\NIVA\SEM 4\ML\LAB\PY FILES\EX 6\churn_boosting.csv")

# encode categorical columns
le = LabelEncoder()
for col in df.select_dtypes(include="object"):
    df[col] = le.fit_transform(df[col])

# split data
X = df.drop("Churn", axis=1)
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# AdaBoost
ada = AdaBoostClassifier(n_estimators=50, random_state=1)
ada.fit(X_train, y_train)
y_pred_ada = ada.predict(X_test)
y_prob_ada = ada.predict_proba(X_test)[:,1]

# Gradient Boosting
gb = GradientBoostingClassifier(n_estimators=50, random_state=1)
gb.fit(X_train, y_train)
y_pred_gb = gb.predict(X_test)
y_prob_gb = gb.predict_proba(X_test)[:,1]

# accuracy
acc_ada = accuracy_score(y_test, y_pred_ada)
acc_gb = accuracy_score(y_test, y_pred_gb)

print("AdaBoost Accuracy:", acc_ada)
print("Gradient Boosting Accuracy:", acc_gb)

# ROC Curve
fpr_ada, tpr_ada, _ = roc_curve(y_test, y_prob_ada)
fpr_gb, tpr_gb, _ = roc_curve(y_test, y_prob_gb)

plt.plot(fpr_ada, tpr_ada, label="AdaBoost")
plt.plot(fpr_gb, tpr_gb, label="Gradient Boosting")
plt.plot([0,1], [0,1])
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("ROC Curve")
plt.legend()
plt.show()

# Feature Importance (Gradient Boosting)
importance = gb.feature_importances_
features = X.columns

plt.barh(features, importance)
plt.title("Feature Importance")
plt.show()