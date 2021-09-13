#rohit sharma www.idexterous.com
import pandas as pd
from sklearn.cross_validation import train_test_split
import Recommenders as Recommenders
import numpy as np


df =  pd.read_csv('data_product.csv')

print(df)

df['cust'] = df['CustomerID'].map(str) + " - " + df['StockCode']

print(df)

cust_grouped = df.groupby(['CustomerID']).agg({'StockCode': 'count'}).reset_index()
grouped_sum = cust_grouped['StockCode'].sum()
cust_grouped['percentage']  = cust_grouped['StockCode'].div(grouped_sum)*100
cust_grouped.sort_values(['StockCode', 'CustomerID'], ascending = [0,1])

users = df['CustomerID'].unique()

print(len(users))

###Fill in the code here
product = df['StockCode'].unique()
print(len(product))


train_data, test_data = train_test_split(df, test_size = 0.20, random_state=0)
print(train_data.head(5))


pm = Recommenders.popularity_recommender_py()
pm.create(train_data, 'CustomerID', 'StockCode')

#users
#user_id replace with CustomerID


user_id = users[5]
pm.recommend(user_id)

###Fill in the code here
user_id = users[8]
pm.recommend(user_id)


is_model = Recommenders.item_similarity_recommender_py()
is_model.create(train_data, 'CustomerID', 'StockCode')

#Print the songs for the user in training data
user_id = users[5]
user_items = is_model.get_user_items(user_id)
#
print("------------------------------------------------------------------------------------")
print("Training data songs for the user userid: %s:" % user_id)
print("------------------------------------------------------------------------------------")

for user_item in user_items:
    print(user_item)

print("----------------------------------------------------------------------")
print("Recommendation process going on:")
print("----------------------------------------------------------------------")

#Recommend songs for the user using personalized model
is_model.recommend(user_id)










