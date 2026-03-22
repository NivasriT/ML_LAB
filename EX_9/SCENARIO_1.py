import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error, mean_absolute_error
print("NIVASRI T | 24BAD081")
# load dataset
df = pd.read_csv(r"C:\Users\nivas\Downloads\archive (9)\ratings.csv")

# check data
print(df.head())
print(df.isnull().sum())

# create user-item matrix
user_item = df.pivot_table(index='userId', columns='movieId', values='rating')

# fill missing values with 0
user_item = user_item.fillna(0)

# show matrix
print(user_item.head())

# plot heatmap
plt.imshow(user_item, aspect='auto')
plt.title("User-Item Matrix")
plt.colorbar()
plt.show()

# compute similarity between users
similarity = cosine_similarity(user_item)

# convert to dataframe
sim_df = pd.DataFrame(similarity, index=user_item.index, columns=user_item.index)

# plot similarity matrix
plt.imshow(sim_df, aspect='auto')
plt.title("User Similarity Matrix")
plt.colorbar()
plt.show()

# function to get top similar users
def get_similar_users(user_id, n=5):
    return sim_df[user_id].sort_values(ascending=False)[1:n+1]

# select one user
user_id = user_item.index[0]

print("Similar Users:\n", get_similar_users(user_id))

# predict ratings
def predict_ratings(user_id):
    similar_users = get_similar_users(user_id).index
    return user_item.loc[similar_users].mean()

# recommend movies
def recommend_movies(user_id, n=5):
    pred = predict_ratings(user_id)
    already_rated = user_item.loc[user_id]
    pred = pred[already_rated == 0]
    return pred.sort_values(ascending=False).head(n)

print("Recommended Movies:\n", recommend_movies(user_id))

# evaluation
pred_matrix = np.dot(similarity, user_item) / np.array([np.abs(similarity).sum(axis=1)]).T

actual = user_item.values.flatten()
predicted = pred_matrix.flatten()

rmse = np.sqrt(mean_squared_error(actual, predicted))
mae = mean_absolute_error(actual, predicted)

print("RMSE:", rmse)
print("MAE:", mae)