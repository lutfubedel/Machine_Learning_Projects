from sklearn.cluster import KMeans
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
import plotly.express as px 
import plotly

# Veriyi okuyoruz
df = pd.read_csv("data.csv")

# Kullanmayacağımız sütunları veri setinden çıkarıyoruz
df = df.drop(['PdDistrict', 'Address', 'Resolution', 'Descript', 'DayOfWeek'], axis=1)

# Yıl bilgisini almak için 'Dates' sütununu işliyoruz
f = lambda x: (x["Dates"].split())[0] 
df["Dates"] = df.apply(f, axis=1)

f = lambda x: (x["Dates"].split('-'))[0] 
df["Dates"] = df.apply(f, axis=1)

# 2014 yılına ait verileri filtreliyoruz ve kopyasını oluşturuyoruz
df_2014 = df[(df.Dates == '2014')].copy()

# Verileri ölçeklendirme işlemi yapıyoruz
scaler = MinMaxScaler()

# X: Boylam (Longitude), Y: Enlem (Latitude)
# Dünya üzerindeki herhangi bir yer enlem ve boylam değerleri ile ifade edilir.

scaler.fit(df_2014[['X']])
df_2014.loc[:, 'X_scaled'] = scaler.transform(df_2014[['X']]) 

scaler.fit(df_2014[['Y']])
df_2014.loc[:, 'Y_scaled'] = scaler.transform(df_2014[['Y']])

# Optimum küme sayısını belirlemek için "Elbow Method" kullanıyoruz
k_range = range(1, 15)
list_dist = []

for k in k_range:
    model = KMeans(n_clusters=k)
    model.fit(df_2014[['X_scaled', 'Y_scaled']])
    list_dist.append(model.inertia_)

# Distortion (inertia) değerlerini çizdiriyoruz
plt.xlabel('K')
plt.ylabel('Distortion Değeri (inertia)')
plt.plot(k_range, list_dist)
plt.show()

# K = 5 için K-Means modelimizi oluşturuyoruz
model = KMeans(n_clusters=5)
y_predicted = model.fit_predict(df_2014[['X_scaled', 'Y_scaled']])

df_2014.loc[:, 'cluster'] = y_predicted

# Harita stili
map_style = "open-street-map"  

# Harita üzerine kümeleri çizdiriyoruz
figure = px.scatter_mapbox(df_2014, 
                           lat='Y', lon='X',                       
                           zoom=9,                           
                           opacity=0.9,                         
                           mapbox_style=map_style,          
                           color='cluster',                    
                           title='San Francisco Suç Bölgeleri',
                           width=1100,
                           height=700,                      
                           hover_data=['cluster', 'Category', 'Y', 'X']  
                           )

figure.show()

# Haritayı HTML olarak kaydedip açıyoruz
plotly.offline.plot(figure, filename='maptest.html', auto_open=True)
