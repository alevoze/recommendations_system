import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from surprise import KNNWithMeans
from surprise import Dataset
from surprise import accuracy
from surprise import Reader
from surprise.model_selection import train_test_split
import warnings; warnings.simplefilter('ignore')
from sklearn.decomposition import TruncatedSVD
plt.show()

import psycopg2




#Baca dan ubah dataset ke Dataframe
electronic_df = pd.read_csv('ratings_Electronics.csv', names=['userId', 'productId', 'Rating', 'timestamp'])
electronic_df.head()

# cek shape data
electronic_df.shape
# cek subset dari dataset
electronic_df = electronic_df.iloc[:1048576,0:]

# check missing value
print('Jumlah value yang hilang pada kolom: \n', electronic_df.isnull().sum())

# Hapus timestamp column
electronic_df.drop(['timestamp'], axis=1, inplace=True)

# Analisi rating
jumlah_rating_produk_per_user = electronic_df.groupby(by='userId')['Rating'].count().sort_values(ascending=False)
jumlah_rating_produk_per_user.head()
# print('\n')

print('\n Produk-produk paling populer berjumlah: {}\n'.format(sum(jumlah_rating_produk_per_user >= 50)) )

# Ambil dataframe baru yang berkaitan dengan user yang memberi 50 rating atau lebih
new_df = electronic_df.groupby("productId").filter(lambda x:x['Rating'].count() >= 50)

# Collaborative
# Read dataset
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(new_df, reader)

# Split dataframe
trainset, testset = train_test_split(data, test_size=0.3, random_state=10)

# Use user-based boolean untuk switch antara CF item-based atau user-based
algo = KNNWithMeans(k=5, sim_options={'name':'pearson_baseline', 'user_based':False})
# Lakukan train
algo.fit(trainset)
test_pred = algo.test(testset)

#get RSME
print("Item-based Model: Test Set")
accuracy.rmse(test_pred, verbose=True)

new_df1=new_df.head(10000)
ratings_matrix = new_df1.pivot_table(values='Rating', index='userId', columns='productId', fill_value=0)
ratings_matrix.head(100)

ratings_matrix.shape

# Transposing the matrix

X = ratings_matrix.T
X.head()

# X = ratings_matrix
# X.head()
X.shape


X1 = X

#Decomposing the Matrix

SVD = TruncatedSVD(n_components=10)
decomposed_matrix = SVD.fit_transform(X)
decomposed_matrix.shape

#Correlation Matrix

correlation_matrix = np.corrcoef(decomposed_matrix)
correlation_matrix.shape

X.index[75]

# Index of product ID purchased by customer

i = "B00000K135"

product_names = list(X.index)
product_ID = product_names.index(i)
product_ID

correlation_product_ID = correlation_matrix[product_ID]
correlation_product_ID.shape

# Recommending top 25 highly correlated products in sequence

Recommend = list(X.index[correlation_product_ID > 0.70])

# Removes the item already bought by the customer
Recommend.remove(i)

data_rec = Recommend[0:100]



conn = psycopg2.connect("host=localhost dbname=postgres user=postgres password=alevoze")
cur = conn.cursor()
cur.execute(
"""
    CREATE TABLE recommendations(
    result_rec TEXT []
)
"""
)
# query = "INSERT INTO recommendations(result_rec) VALUES (%s) (data_rec,)",
cur.execute("INSERT INTO recommendations(result_rec) VALUES (%s)", [data_rec]);
conn.commit()
