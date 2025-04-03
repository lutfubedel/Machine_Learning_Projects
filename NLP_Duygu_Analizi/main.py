import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from bs4 import BeautifulSoup
import re
import nltk
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords

# Veri setini yüklüyoruz (TSV formatında, tab ile ayrılmış).
df = pd.read_csv('data.tsv', delimiter="\t", quoting=3)

# NLTK kütüphanesinden İngilizce stopwords kelime listesini indiriyoruz.
nltk.download('stopwords')

# Örnek bir incelemeyi alıyoruz.
sample_review = df.review[0]

# Metni temizleme fonksiyonu
def process(review):
    # HTML etiketlerini temizleme
    review = BeautifulSoup(review, "html.parser").get_text()
    # Harf dışındaki karakterleri kaldırma (sayılar ve noktalama işaretleri)
    review = re.sub("[^a-zA-Z]", ' ', review)
    # Küçük harfe çevirme ve kelimelere ayırma
    review = review.lower()
    review = review.split()
    # Stopwords listesini yükleme
    swords = set(stopwords.words("english"))  # Set olarak tanımlamak arama işlemini hızlandırır.
    # Stopwords olmayan kelimeleri filtreleme
    review = [w for w in review if w not in swords]               
    # Kelimeleri tekrar birleştirme ve sonucu döndürme
    return " ".join(review)

# Tüm verileri temizleyerek işliyoruz ve ilerleme durumunu her 1000 yorumda bir yazdırıyoruz.
train_x_tum = []
for r in range(len(df["review"])):        
    if (r+1) % 1000 == 0:        
        print("İşlenen yorum sayısı =", r+1)
    train_x_tum.append(process(df["review"][r]))

# Giriş (X) ve etiket (Y) verilerini belirleme
x = train_x_tum
y = np.array(df["sentiment"])

# Eğitim ve test verilerini ayırma (verilerin %10'u test için ayrılıyor)
train_x, test_x, y_train, y_test = train_test_split(x, y, test_size=0.1)

# CountVectorizer ile en fazla 5000 kelimelik "bag of words" modeli oluşturuyoruz.
vectorizer = CountVectorizer(max_features=5000)

# Eğitim verilerini kelime frekans matrisi haline getiriyoruz.
train_x = vectorizer.fit_transform(train_x)

# Matris formatını numpy array'e çeviriyoruz (çünkü model bu formatı bekliyor).
train_x = train_x.toarray()
train_y = y_train

# Rastgele Orman (Random Forest) sınıflandırıcısını eğitiyoruz.
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(train_x, train_y)

# Test verilerini de aynı şekilde kelime frekans matrisine çeviriyoruz.
test_xx = vectorizer.transform(test_x)
test_xx = test_xx.toarray()

# Model ile tahmin yapıyoruz.
test_predict = model.predict(test_xx)

# ROC-AUC skoru ile doğruluk oranını hesaplıyoruz.
dogruluk = roc_auc_score(y_test, test_predict)

print("Doğruluk oranı: %", dogruluk * 100)
