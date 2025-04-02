import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Veri setini yükle
url = "pca_iris.data"
df = pd.read_csv(url, names=['sepal length', 'sepal width', 'petal length', 'petal width', 'target'])

# Kullanılacak özellikler
features = ['sepal length', 'sepal width', 'petal length', 'petal width']

# X: Özellikler, Y: Hedef değişken
x = df[features]
y = df[['target']]

# Özellikleri ölçeklendir (standartlaştır)
x = StandardScaler().fit_transform(x)

# PCA ile boyut azaltma (2 bileşene indir)
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)

# PCA sonuçlarını DataFrame'e çevir
principalDf = pd.DataFrame(data=principalComponents, columns=['principal component 1', 'principal component 2'])

# Orijinal veri setindeki hedef değişken ile birleştir
final_dataframe = pd.concat([principalDf, df[['target']]], axis=1)

# Sınıflar ve renkler
targets = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
colors = ['g', 'b', 'r']

# PCA sonuçlarını görselleştir
plt.figure(figsize=(8, 6))
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA: Iris Dataset')

# Her sınıfı farklı renkte göster
for target, col in zip(targets, colors):
    dftemp = final_dataframe[df.target == target]
    plt.scatter(dftemp['principal component 1'], dftemp['principal component 2'], color=col, label=target)

# Grafik için legend ekle ve göster
plt.legend()
plt.show()
