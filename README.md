# Pusula_Edanur_Tabu

#Edanur Tabu
tabuedanur@gmail.com





# Tedavi Süresi Raporlama ve Analiz Projesi

Bu proje, bir hastane veri seti üzerinden **hasta tedavi sürelerini analiz edip raporlamak** için hazırlanmıştır. 
PDF formatında görsel ve tablo tabanlı bir rapor oluşturulmaktadır. Rapor, hasta verilerindeki dağılımlar, cinsiyet ve bölüm analizi, tedavi süresi tahmini ve model performansını içerir.

---

## Proje Dosyaları

- `veri.xlsx` : Hasta verilerini içeren Excel dosyası
- `rapor.py` : Raporlama ve analiz pipeline kodu
- `Tedavi_Rapor_Analizli.pdf` : Oluşturulan PDF rapor
- `README.md` : Proje açıklamaları ve kullanım talimatları

---
## PDF Raporu

📄 **Raporu Görüntüle:**  
[Raporu Aç](Tedavi_Rapor_Analizli.pdf)  

PDF içinde şunlar yer alır:  
- Yaş dağılımı  
- Tedavi süresi dağılımı  
- Cinsiyet ve bölüm dağılımları  
- Gerçek vs tahmin grafiği  
- Model performans tablosu  
- Analiz ve yorumlar

## Gereksinimler

Python 3.12 veya üzeri ve aşağıdaki kütüphaneler gereklidir:

```bash
pip install pandas matplotlib seaborn scikit-learn openpyxl
