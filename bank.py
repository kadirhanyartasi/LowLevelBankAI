import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
#kütüphanelerimizi ekliyoruz ve proje ana amacımız kullanıcının data sete göre kart kayıp,paragönderme vb. işlemlerini otomatizeye calısıyoruz

df=pd.read_csv("banka.csv") #dosyamızı acıyoruz
mesaj=input("Yapmak İstediniz İşlemi Giriniz.")#işlemi alıyoruz

mesajdf=pd.DataFrame({"sorgu":mesaj,"label":0},index=[42])#işlemi data frame e gönderiyoruz

df=pd.concat([df,mesajdf],ignore_index=True)#indexe gerek yok.

stopwords=['fakat','lakin','ancak','acaba', 'ama', 'aslında', 'az', 'bazı', 'belki', 'biri', 'birkaç', 'birşey', 'biz', 'bu', 'çok', 'çünkü', 'da', 'daha', 'de', 'defa', 'diye', 'eğer', 'en', 'gibi', 'hem', 'hep', 'hepsi', 'her', 'hiç', 'için', 'ile', 'ise', 'kez', 'ki', 'kim', 'mı', 'mu', 'mü', 'nasıl', 'ne', 'neden', 'nerde', 'nerede', 'nereye', 'niçin', 'niye', 'o', 'sanki', 'şey', 'siz', 'şu', 'tüm', 've', 'veya', 'ya', 'yani']
#stopwordler sayesinde uzun cümleleri sakinleştirebiliyorz

#bu for döngüsüyle birlikte stopwordun kelime olmasını saglıyoruz örn:"gama daki ama yı silmicek."
for word in stopwords:
    word=" "+word+" "
    df['sorgu']=df['sorgu'].str.replace(word," ")

cv=CountVectorizer(max_features=50)#vektörlüyoruz

#
x=cv.fit_transform(df['sorgu']).toarray()
y=df['label']

#tahmin
tahmin=x[-1].copy()

x=x[0:-1]
y=y[0:-1]
#egitim
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=21,train_size=0.7)

rf=RandomForestClassifier()
model=rf.fit(x_train,y_train)
skor=model.score(x_test,y_test)

sonuc=model.predict([tahmin])
#sonuc yazdırma
print("Sonuc:",sonuc,"Skor:",skor)

