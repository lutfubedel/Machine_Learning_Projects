'''
    Machine Learning modellerinden KNN Modeli ile Python'da seker hastaligi tespiti uygulamasi 
'''

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler

# Veriseti yuklenir
data = pd.read_csv("diabetes.csv")

# Seker hastalari ve saglikli bireyler ayrı ayrı degiskenlerde tutulur
seker_hastalari = data[data.Outcome == 1]
saglikli_insanlar = data[data.Outcome == 0]

# Glucose degerine gore grafik olusturulur.
plt.scatter(saglikli_insanlar.Age, saglikli_insanlar.Glucose, color="green", label="sağlıklı", alpha = 0.4)
plt.scatter(seker_hastalari.Age, seker_hastalari.Glucose, color="red", label="diabet hastası", alpha = 0.4)
plt.xlabel("Age")
plt.ylabel("Glucose")
plt.legend()
plt.show()

# Verisetini y : bagimli degisken x : bagimsiz degiskenler olacak sekilde ayrilir.
y = data.Outcome.values
x_ham_veri = data.drop(["Outcome"],axis=1)  

# Bagimsiz degiskenlere normalizasyon uygulanir.
x = (x_ham_veri - np.min(x_ham_veri))/(np.max(x_ham_veri)-np.min(x_ham_veri))

# x ve y verisetleri train ve test olmak uzere 2 ye ayrilir.
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.1,random_state=1)

# KNN modeli
knn = KNeighborsClassifier(n_neighbors = 3) # n_neighbors = k
knn.fit(x_train,y_train)
prediction = knn.predict(x_test)

print("K=3 için Test verilerimizin doğrulama testi sonucu ", knn.score(x_test, y_test))

# Yeni veriyi DataFrame olarak oluştur ve aynı normalizasyonu uygula
new_data = pd.DataFrame(np.array([[6, 148, 72, 35, 0, 33.6, 0.627, 50]]), columns=x_ham_veri.columns)
new_data_normalized = (new_data - x_ham_veri.min()) / (x_ham_veri.max() - x_ham_veri.min())

# Tahmin yap
new_prediction = knn.predict(new_data_normalized)

print("Yeni hastanın tahmini (0: Sağlıklı, 1: Diyabet Hastası):", new_prediction[0])