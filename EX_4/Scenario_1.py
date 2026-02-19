print("NIVASRI T | 24BAD081")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import string

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Load dataset
data = pd.read_csv(r"C:\Users\nivas\Desktop\NIVA\SEM 4\ML\LAB\PY FILES\ex4\spam.csv", encoding='latin-1')
data = data[['v1','v2']]
data.columns = ['label','message']

# Text cleaning
data['message'] = data['message'].str.lower()
data['message'] = data['message'].str.replace(f"[{string.punctuation}]", "", regex=True)

# Encode labels
data['label'] = data['label'].map({'ham':0, 'spam':1})

# Vectorization
vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(data['message'])
y = data['label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = MultinomialNB(alpha=1.0)
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure()
sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
# Feature Importance (Top Spam Words)
feature_names = np.array(vectorizer.get_feature_names_out())
spam_log_prob = model.feature_log_prob_[1]
top_spam = feature_names[np.argsort(spam_log_prob)[-10:]]

plt.figure()
plt.barh(top_spam, np.sort(spam_log_prob)[-10:])
plt.title("Top Words Influencing Spam")
plt.show()
# Word Frequency Comparison
spam_words = data[data['label']==1]['message']
ham_words = data[data['label']==0]['message']

spam_counts = CountVectorizer(stop_words='english').fit_transform(spam_words)
ham_counts = CountVectorizer(stop_words='english').fit_transform(ham_words)

spam_sum = np.array(spam_counts.sum(axis=0)).flatten()
ham_sum = np.array(ham_counts.sum(axis=0)).flatten()

spam_features = CountVectorizer(stop_words='english').fit(spam_words).get_feature_names_out()
ham_features = CountVectorizer(stop_words='english').fit(ham_words).get_feature_names_out()

top_spam_freq = spam_features[np.argsort(spam_sum)[-10:]]
top_ham_freq = ham_features[np.argsort(ham_sum)[-10:]]

plt.figure()
plt.barh(top_spam_freq, np.sort(spam_sum)[-10:])
plt.title("Most Frequent Words in Spam")
plt.show()

plt.figure()
plt.barh(top_ham_freq, np.sort(ham_sum)[-10:])
plt.title("Most Frequent Words in Ham")
plt.show()
