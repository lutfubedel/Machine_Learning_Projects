import numpy as np
import pandas as pd 

from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
from kmodes.kprototypes import KPrototypes  
from kmodes.kmodes import KModes

# Veriyi oku
df = pd.read_csv("data.csv")

# 'ID', 'Age' ve 'Income' sütunlarını geçici olarak sakla
df_temp = df[['ID', 'Age', 'Income']]

# Min-Max ölçeklendirme için nesneyi oluştur
scaler = MinMaxScaler()

# 'Age' sütununu ölçekle
scaler.fit(df[['Age']])
df['Age'] = scaler.transform(df[['Age']])

# 'Income' sütununu ölçekle
scaler.fit(df[['Income']])
df['Income'] = scaler.transform(df[['Income']])

# 'ID' sütununu veri setinden çıkar
df = df.drop(['ID'], axis=1)

# Veriyi numpy array formatına çevir
mark_array = df.values

# Belirtilen sütunları float türüne çevir
mark_array[:, 2] = mark_array[:, 2].astype(float)
mark_array[:, 4] = mark_array[:, 4].astype(float)

# K-Prototypes modelini oluştur ve eğit
kproto = KPrototypes(n_clusters=10, verbose=2, max_iter=20)
clusters = kproto.fit_predict(mark_array, categorical=[0, 1, 3, 5, 6])

# Küme merkezlerini yazdır
print(kproto.cluster_centroids_)
print(len(kproto.cluster_centroids_))

# Küme etiketlerini listeye kaydet
cluster_dict = []
for c in clusters:
    cluster_dict.append(c)

# Küme bilgilerini veri çerçevesine ekle
df['cluster'] = cluster_dict

# Geçici olarak saklanan sütunları geri ekle
df[['ID', 'Age', 'Income']] = df_temp

# Her kümeyi ayrı veri çerçevelerine ayır
df1 = df[df.cluster == 0]
df2 = df[df.cluster == 1]
df3 = df[df.cluster == 2]
df4 = df[df.cluster == 3]
df5 = df[df.cluster == 4]
df6 = df[df.cluster == 6]
df7 = df[df.cluster == 7]
df8 = df[df.cluster == 8]
df9 = df[df.cluster == 9]
df10 = df[df.cluster == 10]  

# Grafik oluştur
plt.figure(figsize=(15, 15))
plt.xlabel('Age')
plt.ylabel('Income')

# Her kümeyi farklı renklerde görselleştir
plt.scatter(df1.Age, df1['Income'], color='green', alpha=0.4)
plt.scatter(df2.Age, df2['Income'], color='red', alpha=0.4)
plt.scatter(df3.Age, df3['Income'], color='gray', alpha=0.4)
plt.scatter(df4.Age, df4['Income'], color='orange', alpha=0.4)
plt.scatter(df5.Age, df5['Income'], color='yellow', alpha=0.4)
plt.scatter(df6.Age, df6['Income'], color='cyan', alpha=0.4)
plt.scatter(df7.Age, df7['Income'], color='magenta', alpha=0.4)
plt.scatter(df8.Age, df8['Income'], color='gray', alpha=0.4)
plt.scatter(df9.Age, df9['Income'], color='purple', alpha=0.4)
plt.scatter(df10.Age, df10['Income'], color='blue', alpha=0.4) 

# Grafiği göster
plt.legend()
plt.show()
