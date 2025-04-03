import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_openml  # MNIST veri setini yüklemek için
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# MNIST veri setini yükleyelim (Bu işlem 1-2 dk sürebilir)
mnist = fetch_openml('mnist_784')

# Parametre olarak dataframe ve ilgili veri fotoğrafının index numarasını alan fonksiyon
def showimage(dframe, index):    
    some_digit = dframe.to_numpy()[index]  # Belirtilen indexteki veriyi al
    some_digit_image = some_digit.reshape(28,28)  # 28x28 formatına getir
    
    plt.imshow(some_digit_image, cmap="binary")  # Siyah-beyaz göster
    plt.axis("off")  # Eksenleri kapat
    plt.show()

# Test ve train oranı: %85 (train), %15 (test)
train_img, test_img, train_lbl, test_lbl = train_test_split(mnist.data, mnist.target, test_size=1/7.0, random_state=0)

# Test setini kaybetmemek için bir kopyasını alıyoruz
test_img_copy = test_img.copy()

# Veriyi ölçeklendirmek için StandardScaler kullanıyoruz
scaler = StandardScaler()

# Scaler'ı sadece eğitim verisi üzerinde eğitiyoruz
scaler.fit(train_img)

# Hem eğitim hem de test verisine dönüşüm uyguluyoruz
train_img = scaler.transform(train_img)
test_img = scaler.transform(test_img)

# PCA modelini oluşturuyoruz ve %95 varyansı koruyacak şekilde ayarlıyoruz
pca = PCA(.95)

# PCA'i sadece eğitim verisi üzerinde fit ediyoruz (1 dk sürebilir)
pca.fit(train_img)

# PCA ile veri boyutunun ne kadar küçüldüğünü kontrol edelim
print("PCA sonrası bileşen sayısı:", pca.n_components_)

# Hem eğitim hem de test verisini PCA ile dönüştürüyoruz
train_img = pca.transform(train_img)
test_img = pca.transform(test_img)

# Logistic Regression modelini tanımlıyoruz (lbfgs solver'ı ile daha hızlı çalışır)
logisticRegr = LogisticRegression(solver='lbfgs', max_iter=10000)

# Modeli eğitiyoruz (Birkaç dakika sürebilir)
logisticRegr.fit(train_img, train_lbl)

# İlk test verisini kullanarak tahminde bulunuyoruz
prediction = logisticRegr.predict(test_img[0].reshape(1, -1))
print("Tahmin edilen rakam:", prediction[0])

# Modelin doğruluk oranını ölçüyoruz
score = logisticRegr.score(test_img, test_lbl)
print("Modelin doğruluk oranı:", score)