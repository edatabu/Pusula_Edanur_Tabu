import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder


df = pd.read_excel("veri.xlsx")
print("veri boyutu:" , df.shape)
print(df.head()) 
print(df.info())
print(df.describe())

print("\nEksik Değer Sayıları:")
print(df.isnull().sum())


df["TedaviSuresi_sayi"] = df["TedaviSuresi"].str.extract("(\d+)").astype(float)

plt.figure(figsize=(8,6))
sns.histplot(df["TedaviSuresi_sayi"], bins=20, kde=True)
plt.title("Tedavi Süresi Dağılımı")
plt.show()


sns.histplot(df["Yas"], bins=20, kde=True)
plt.title("Yaş Dağılımı")
plt.show()

sns.boxplot(x=df["TedaviSuresi_sayi"])
plt.title("Tedavi Süresi Kutu Grafiği")
plt.show()


print(df["Cinsiyet"].value_counts())
sns.countplot(x="Cinsiyet", data=df)
plt.title("Cinsiyet Dağılımı")
plt.show()

print(df["Bolum"].value_counts())
plt.figure(figsize=(10,5))
sns.countplot(x="Bolum", data=df, order=df["Bolum"].value_counts().index)
plt.xticks(rotation=45)
plt.title("Bölüm Dağılımı")
plt.show()

plt.figure(figsize=(10,6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Korelasyon Matrisi")
plt.show()


sns.scatterplot(x="Yas", y="TedaviSuresi_sayi", data=df)
plt.title("Yaş ve Tedavi Süresi İlişkisi")
plt.show()

sns.boxplot(x="Cinsiyet", y="TedaviSuresi_sayi", data=df)
plt.title("Cinsiyet ve Tedavi Süresi")
plt.show()

df["Cinsiyet"] = df["Cinsiyet"].fillna("Bilinmiyor")
df["KanGrubu"] = df["KanGrubu"].fillna("Bilinmiyor") 
df["KronikHastalik"] = df["KronikHastalik"].fillna("Yok")
df["Alerji"] = df["Alerji"].fillna("Yok")
df["Tanilar"] = df["Tanilar"].fillna("Bilinmiyor")
df["UygulamaYerleri"] = df["UygulamaYerleri"].fillna("Belirtilmemiş")


df["Yas"] = df["Yas"].astype(int)
df["TedaviSuresi_sayi"] = df["TedaviSuresi_sayi"].astype(int)


df["Cinsiyet"] = df["Cinsiyet"].str.strip().str.capitalize()
df["Uyruk"] = df["Uyruk"].str.strip().str.capitalize()


print("Tekrar eden satır sayısı:", df.duplicated().sum())
df = df.drop_duplicates()


sns.boxplot(x=df["Yas"])
plt.show()

sns.boxplot(x=df["TedaviSuresi_sayi"])
plt.show()


Q1 = df["TedaviSuresi_sayi"].quantile(0.25)
Q3 = df["TedaviSuresi_sayi"].quantile(0.75)
IQR = Q3 - Q1
alt_sinir = Q1 - 1.5 * IQR
ust_sinir = Q3 + 1.5 * IQR

df = df[(df["TedaviSuresi_sayi"] >= alt_sinir) & (df["TedaviSuresi_sayi"] <= ust_sinir)]


Q1 = df["Yas"].quantile(0.25)
Q3 = df["Yas"].quantile(0.75)
IQR = Q3 - Q1
alt_sinir = Q1 - 1.5 * IQR
ust_sinir = Q3 + 1.5 * IQR

df = df[(df["Yas"] >= alt_sinir) & (df["Yas"] <= ust_sinir)]

cat_cols = ["Cinsiyet", "KanGrubu", "Uyruk", "Bolum"]

df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df[["Yas", "TedaviSuresi_sayi"]] = scaler.fit_transform(df[["Yas", "TedaviSuresi_sayi"]])
print("Son veri boyutu:", df.shape)
print(df.head())
print(df.info())

X = df.drop("TedaviSuresi_sayi", axis=1)  
y = df["TedaviSuresi_sayi"]             
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Eğitim boyutu:", X_train.shape)
print("Test boyutu:", X_test.shape)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))
plt.scatter(y_test, y_pred)
plt.xlabel("Gerçek Tedavi Süresi")
plt.ylabel("Tahmin Edilen Tedavi Süresi")
plt.title("Gerçek vs Tahmin Edilen Tedavi Süresi")
plt.show()
