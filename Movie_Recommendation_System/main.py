import numpy as np
import pandas as pd

# Kullanıcı verilerini yükleyelim (user_id, item_id, rating, timestamp)
column_names = ['user_id', 'item_id', 'rating', 'timestamp']
df = pd.read_csv('users.data', sep='\t', names=column_names)

# Film ID'lerine karşılık gelen film isimlerini içeren CSV dosyasını yükleyelim
movie_titles = pd.read_csv("movie_id_titles.csv")

# Film ID'sine göre iki veri çerçevesini birleştirelim
df = pd.merge(df, movie_titles, on='item_id')

# Kullanıcı-film matrisini oluşturalım (pivot table yöntemi ile)
moviemat = df.pivot_table(index='user_id', columns='title', values='rating')

# Star Wars (1977) filmini izleyen kullanıcıların puanlarını alalım
starwars_user_ratings = moviemat['Star Wars (1977)']

# Star Wars (1977) ile diğer filmler arasındaki korelasyonu hesaplayalım
similar_to_starwars = moviemat.corrwith(starwars_user_ratings)

# Korelasyonları DataFrame formatına dönüştürelim
corr_starwars = pd.DataFrame(similar_to_starwars, columns=['Correlation'])
corr_starwars.dropna(inplace=True)  # Boş değerleri kaldıralım

# Korelasyon değerlerini büyükten küçüğe sıralayarak en benzer 10 filmi görelim
corr_starwars.sort_values('Correlation', ascending=False).head(10)

# Gereksiz timestamp sütununu kaldırıyoruz
df.drop(['timestamp'], axis=1, inplace=True)

# Her filmin ortalama (mean) rating değerini hesaplayalım
ratings = pd.DataFrame(df.groupby('title')['rating'].mean())

# Ortalama puanlara göre büyükten küçüğe sıralayarak ilk 5 filmi listeleyelim
ratings.sort_values('rating', ascending=False).head()

# Her filmin kaç kez oylandığını hesaplayalım
ratings['rating_oy_sayisi'] = df.groupby('title')['rating'].count()

# En çok oy alan 5 filmi listeleyelim
ratings.sort_values('rating_oy_sayisi', ascending=False).head()

# Korelasyon tablosuna oy sayısını ekleyelim
corr_starwars = corr_starwars.join(ratings['rating_oy_sayisi'])

# Yeterince fazla oy almış (örneğin, 100'den fazla) filmler arasından en yüksek korelasyonlu olanları sıralayalım
print(corr_starwars[corr_starwars['rating_oy_sayisi'] > 100].sort_values('Correlation', ascending=False).head())