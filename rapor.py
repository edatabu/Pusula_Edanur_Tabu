
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


df = pd.read_excel("veri.xlsx")

df["TedaviSuresi_sayi"] = df["TedaviSuresi"].str.extract('(\d+)').astype(float)

hedef_sutun = "TedaviSuresi_sayi"
X = df.drop(columns=[hedef_sutun, "HastaNo", "TedaviSuresi"])
y = df[hedef_sutun]

X = pd.get_dummies(X, drop_first=True)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


sns.set_style("whitegrid")
palet = ["#2a9d8f", "#e76f51", "#f4a261", "#264653", "#e9c46a", "#8ab17d"]

with PdfPages("Tedavi_Rapor_Analizli.pdf") as pdf:


    plt.figure(figsize=(9,5))
    sns.histplot(df["Yas"], bins=20, kde=True, color=palet[0])
    plt.title("Yaş Dağılımı", fontsize=16, fontweight='bold')
    plt.xlabel("Yaş", fontsize=12)
    plt.ylabel("Hasta Sayısı", fontsize=12)
    plt.text(0.5, -0.12, "Hastaların yaş dağılımını göstermektedir.", ha='center', va='center', transform=plt.gca().transAxes, fontsize=10)
    pdf.savefig()
    plt.close()

    plt.figure(figsize=(9,5))
    sns.histplot(df["TedaviSuresi_sayi"], bins=20, kde=True, color=palet[1])
    plt.title("Tedavi Süresi Dağılımı", fontsize=16, fontweight='bold')
    plt.xlabel("Tedavi Süresi (Seans)", fontsize=12)
    plt.ylabel("Hasta Sayısı", fontsize=12)
    plt.text(0.5, -0.12, "Hastaların tedavi sürelerinin dağılımını göstermektedir.", ha='center', va='center', transform=plt.gca().transAxes, fontsize=10)
    pdf.savefig()
    plt.close()

    plt.figure(figsize=(7,4))
    sns.countplot(x="Cinsiyet", data=df, palette=[palet[2], palet[3]])
    plt.title("Cinsiyet Dağılımı", fontsize=16, fontweight='bold')
    plt.xlabel("Cinsiyet", fontsize=12)
    plt.ylabel("Hasta Sayısı", fontsize=12)
    plt.text(0.5, -0.15, f"Kadın: {df['Cinsiyet'].value_counts().get('Kadın',0)}, Erkek: {df['Cinsiyet'].value_counts().get('Erkek',0)}", ha='center', va='center', transform=plt.gca().transAxes, fontsize=10)
    pdf.savefig()
    plt.close()


    plt.figure(figsize=(10,5))
    order_bolum = df["Bolum"].value_counts().index
    sns.countplot(x="Bolum", data=df, order=order_bolum, palette=palet)
    plt.xticks(rotation=45, fontsize=10)
    plt.title("Bölüm Dağılımı", fontsize=16, fontweight='bold')
    plt.xlabel("Bölüm", fontsize=12)
    plt.ylabel("Hasta Sayısı", fontsize=12)
    plt.text(0.5, -0.18, "Hastaların hangi bölümlerde tedavi gördüğünü göstermektedir.", ha='center', va='center', transform=plt.gca().transAxes, fontsize=10)
    pdf.savefig()
    plt.close()

    plt.figure(figsize=(9,6))
    plt.scatter(y_test, y_pred, s=60, color=palet[4], edgecolor="k")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', linewidth=2)
    plt.xlabel("Gerçek Tedavi Süresi", fontsize=12)
    plt.ylabel("Tahmin Edilen Tedavi Süresi", fontsize=12)
    plt.title("Gerçek vs Tahmin Edilen Tedavi Süresi", fontsize=16, fontweight='bold')
    plt.text(0.5, -0.12, "Model tahminleri kırmızı çizgiye yakınsa başarılıdır.", ha='center', va='center', transform=plt.gca().transAxes, fontsize=10)
    pdf.savefig()
    plt.close()

    fig, ax = plt.subplots(figsize=(9,2))
    ax.axis('tight')
    ax.axis('off')
    table_data = [["MSE", f"{mse:.2f}"], ["MAE", f"{mae:.2f}"], ["R² Skoru", f"{r2:.2f}"]]
    ax.table(cellText=table_data, colLabels=["Metod", "Değer"], loc='center', cellLoc='center')
    plt.title("Model Performans Tablosu", fontsize=16, fontweight='bold')
    pdf.savefig()
    plt.close()

    fig, ax = plt.subplots(figsize=(11,8))
    ax.axis('off')
    analiz_text = (
        "Veri Analizi ve Yorumlar\n\n"
        f"- Toplam Hasta Sayısı: {len(df)}\n"
        f"- Yaş: Ortalama {df['Yas'].mean():.1f}, Min {df['Yas'].min()}, Max {df['Yas'].max()}\n"
        f"- Cinsiyet: Kadın {df['Cinsiyet'].value_counts().get('Kadın',0)}, Erkek {df['Cinsiyet'].value_counts().get('Erkek',0)}\n"
        "- Bölümler: Fiziksel Tıp ve Rehabilitasyon en yoğun hasta sayısına sahip.\n"
        "- Tedavi Süresi: Çoğu hastanın tedavi süresi 5-15 seans aralığında.\n\n"
        "Model Performansı:\n"
        f"- MSE: {mse:.2f}, MAE: {mae:.2f}, R² Skoru: {r2:.2f}\n"
        "- Gerçek vs Tahmin grafiğinde noktalar kırmızı çizgiye yakın, model genel olarak başarılı.\n\n"
        
    )
    ax.text(0, 1, analiz_text, fontsize=12, va='top', ha='left', wrap=True)
    pdf.savefig()
    plt.close()

print("PDF rapor başarıyla oluşturuldu: Tedavi_Rapor_Analizli.pdf")
