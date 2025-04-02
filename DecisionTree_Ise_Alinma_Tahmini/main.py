import pandas as pd
from sklearn import tree

# Veri setini yükle
df = pd.read_csv("data.csv")

# Kategorik verileri sayısala çevir
duzetme_mapping = {'Y': 1, 'N': 0}
df['IseAlindi'] = df['IseAlindi'].map(duzetme_mapping)
df['SuanCalisiyor?'] = df['SuanCalisiyor?'].map(duzetme_mapping)
df['Top10 Universite?'] = df['Top10 Universite?'].map(duzetme_mapping)
df['StajBizdeYaptimi?'] = df['StajBizdeYaptimi?'].map(duzetme_mapping)

duzetme_mapping_egitim = {'BS': 0, 'MS': 1, 'PhD': 2}
df['Egitim Seviyesi'] = df['Egitim Seviyesi'].map(duzetme_mapping_egitim)

# Bağımlı ve bağımsız değişkenleri ayır
y = df['IseAlindi']
X = df.drop(['IseAlindi'], axis=1)

# Modeli eğit
clf = tree.DecisionTreeClassifier()
clf.fit(X, y)

# Yeni verileri bir DataFrame olarak oluştur (X'in sütun isimlerini kullanarak)
yeni_veri1 = pd.DataFrame([[5, 1, 3, 0, 0, 0]], columns=X.columns)
yeni_veri2 = pd.DataFrame([[2, 0, 7, 0, 1, 0]], columns=X.columns)

# Tahmin yap
tahmin1 = clf.predict(yeni_veri1)
tahmin2 = clf.predict(yeni_veri2)

print("Tahmin Sonucu (1. aday):", tahmin1[0])  # 1: işe alındı, 0: alınmadı
print("Tahmin Sonucu (2. aday):", tahmin2[0])  # 1: işe alındı, 0: alınmadı
