import numpy  # NumPy kütüphanesi
import os  # İşletim sistemi işlemleri için

# OpenMP paralel iş parçacığı sayısını 1'e ayarlıyoruz
os.environ["OMP_NUM_THREADS"] = "1"

from sklearn.cluster import KMeans  # K-Means kümeleme algoritması
import pandas as pd  # Veri işleme için
from sklearn.preprocessing import MinMaxScaler  # Veri ölçekleme
from matplotlib import pyplot as plt  # Grafik çizme

# Veriyi CSV dosyasından okuyarak yükleyelim
df = pd.read_csv("data.csv")

# Gelir ve harcama skorunu görselleştirelim
plt.scatter(df['Annual Income (k$)'], df['Spending Score (1-100)'])
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.show()

# Sütun isimlerini daha kısa hale getirelim
df.rename(columns={'Annual Income (k$)': 'income'}, inplace=True)
df.rename(columns={'Spending Score (1-100)': 'score'}, inplace=True)

# Veriyi MinMaxScaler ile ölçekleyelim
scaler = MinMaxScaler()
scaler.fit(df[['income']])
df['income'] = scaler.transform(df[['income']])
scaler.fit(df[['score']])
df['score'] = scaler.transform(df[['score']])

df.head()

# En uygun küme sayısını belirlemek için Elbow yöntemini kullanalım
k_range = range(1, 11)
list_dist = []
for k in k_range:
    kmeans_modelim = KMeans(n_clusters=k)
    kmeans_modelim.fit(df[['income', 'score']])
    list_dist.append(kmeans_modelim.inertia_)

plt.xlabel('K')
plt.ylabel('Distortion değeri (inertia)')
plt.plot(k_range, list_dist)
plt.show()

# K = 5 için model oluşturalım ve küme tahminlerini alalım
kmeans_modelim = KMeans(n_clusters=5)
y_predicted = kmeans_modelim.fit_predict(df[['income', 'score']])
df['cluster'] = y_predicted

# Küme merkezlerini ekrana yazdıralım
print(kmeans_modelim.cluster_centers_)

# Kümeleme sonuçlarını görselleştirelim
df1 = df[df.cluster == 0]
df2 = df[df.cluster == 1]
df3 = df[df.cluster == 2]
df4 = df[df.cluster == 3]
df5 = df[df.cluster == 4]

plt.xlabel('income')
plt.ylabel('score')
plt.scatter(df1['income'], df1['score'], color='green')
plt.scatter(df2['income'], df2['score'], color='red')
plt.scatter(df3['income'], df3['score'], color='black')
plt.scatter(df4['income'], df4['score'], color='orange')
plt.scatter(df5['income'], df5['score'], color='purple')

# Küme merkezlerini de ekleyerek görselleştirelim
plt.scatter(kmeans_modelim.cluster_centers_[:, 0], kmeans_modelim.cluster_centers_[:, 1], color='blue', marker='X', label='centroid')
plt.legend()
plt.show()
