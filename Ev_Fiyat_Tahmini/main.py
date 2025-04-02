import pandas as pd
from sklearn import linear_model

# Multiple Linear Regression 

# Veri setini ice aktar
df = pd.read_csv("multilinearregression.csv", sep=";")

# Linear regression modelini tanimla ve egit
reg = linear_model.LinearRegression()
reg.fit(df[['alan', 'odasayisi', 'binayasi']], df['fiyat'])

# Belirtilen değerler için tahmin yap
predict = reg.predict([[230, 4, 10], [230, 6, 0], [355, 3, 20]])

# Modelin katsayilarini ve sabit terimini yazdir
print("Katsayılar:", reg.coef_)  # Her özelligin katsayisini gosterir
print("Sabit Terim (Intercept):", reg.intercept_)  # Sabit terim

# Tahmin sonuclarini ekrana yazdir
for i, tahmin in enumerate(predict, 1):
    print(f"Tahmin {i}: {tahmin:.2f} TL")
