import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
print("NIVASRI T | 24BAD081")
# load dataset
df = pd.read_csv(r"C:\Users\nivas\Downloads\archive (9)\ratings.csv")

# check data
print(df.head())

# create item-user matrix
item_user = df.pivot_table(index='movieId', columns='userId', values='rating')

# fill missing values
item_user = item_user.fillna(0)

print(item_user.head())

# heatmap
plt.imshow(item_user, aspect='auto')
plt.title("Item-User Matrix")
plt.colorbar()
plt.show()

# compute similarity between items
item_similarity = cosine_similarity(item_user)

sim_df = pd.DataFrame(item_similarity, index=item_user.index, columns=item_user.index)

# similarity heatmap
plt.imshow(sim_df, aspect='auto')
plt.title("Item Similarity Matrix")
plt.colorbar()
plt.show()

# function to get similar items
def get_similar_items(movie_id, n=5):
    return sim_df[movie_id].sort_values(ascending=False)[1:n+1]

# pick one movie
movie_id = item_user.index[0]

print("Similar Items:\n", get_similar_items(movie_id))

# recommend items based on user history
def recommend_items(user_id, n=5):
    user_ratings = item_user[user_id]
    rated_items = user_ratings[user_ratings > 0].index
    
    scores = pd.Series(dtype=float)
    
    for item in rated_items:
        sim_items = sim_df[item]
        scores = scores.add(sim_items * user_ratings[item], fill_value=0)
    
    scores = scores.sort_values(ascending=False)
    return scores.head(n)

# example user
user_id = df['userId'].iloc[0]

print("Recommended Items:\n", recommend_items(user_id))

# evaluation
pred_matrix = np.dot(item_user.T, item_similarity) / np.array([np.abs(item_similarity).sum(axis=1)])

actual = item_user.values.flatten()
predicted = pred_matrix.T.flatten()

rmse = np.sqrt(mean_squared_error(actual, predicted))

print("RMSE:", rmse)