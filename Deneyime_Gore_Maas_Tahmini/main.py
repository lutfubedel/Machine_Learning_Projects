import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Veri setini içe aktar
df = pd.read_csv("data.csv", sep=";")

# Veri setini görselleştir
plt.scatter(df['deneyim'], df['maas'])
plt.xlabel('Deneyim (yıl)')
plt.ylabel('Maaş')
plt.title('Deneyim ve Maaş İlişkisi')
plt.show()

# Lineer Regresyon Modeli
linear_reg = LinearRegression()
linear_reg.fit(df[['deneyim']], df['maas'])  # Modeli eğit

# Lineer Regresyon Sonuçlarını Görselleştir
plt.xlabel('Deneyim (yıl)')
plt.ylabel('Maaş')
plt.scatter(df['deneyim'], df['maas'], label="Gerçek Veriler")  # Orijinal veriler

# Modelin tahminlerini çiz
x_ekseni = df[['deneyim']]
y_ekseni = linear_reg.predict(x_ekseni)
plt.plot(x_ekseni, y_ekseni, color="green", label="Linear Regression")
plt.legend()
plt.title("Lineer Regresyon Sonuçları")
plt.show()

# Polinom Regresyon Modeli (4. dereceden polinom)
polynomial_features = PolynomialFeatures(degree=4)  # 4. dereceden polinom oluştur
x_polynomial = polynomial_features.fit_transform(df[['deneyim']])  # Veriyi polinomal hale getir

# Polinom regresyon modelini eğit
poly_reg = LinearRegression()
poly_reg.fit(x_polynomial, df['maas'])

# Polinom regresyon tahminlerini çiz
y_head = poly_reg.predict(x_polynomial)

plt.scatter(df['deneyim'], df['maas'], label="Gerçek Veriler")  # Orijinal veriler
plt.plot(df['deneyim'], y_head, color="red", label="Polynomial Regression")  # Polinom regresyon çizgisi
plt.plot(x_ekseni, y_ekseni, color="green", label="Linear Regression")  # Lineer regresyon çizgisi
plt.legend()
plt.title("Polinom ve Lineer Regresyon Sonuçları")
plt.show()

# Tahmin yap (Deneyim = 4.5 yıl için maaş tahmini)
x_polynomial1 = polynomial_features.transform([[4.5]])  # Yeni veriyi polinomal hale getir
tahmin = poly_reg.predict(x_polynomial1)  # Tahmini hesapla

# Tahmin sonucunu ekrana yazdır
print(f"Deneyimi 4.5 yıl olan bir kişinin maaş tahmini: {tahmin[0]:.2f} TL")
