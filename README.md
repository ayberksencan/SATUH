# SATUH
SATUH (Sentiment Analysis on Twitter Using Machine Learning, Deep Learning and Hybrid Models

# Twitter Sentiment Analysis

Bu proje, Twitter verilerinden duygu analizi yapmayı amaçlayan yüksek lisans tez çalışmasıdır. Proje, Ömer Ayberk ŞENCAN tarafından Gazi Üniversitesi Fen Bilimleri Enstitüsü Bilgisayar Mühendisliği Bölümü için hazırlanmıştır.

## Proje Özeti
Bu çalışmada, sosyal medya platformlarında üretilen verilerin yapay zekâ temelli modeller ile analiz edilmesi amaçlanmıştır. Twitter kullanıcılarının ABD'deki altı büyük havayolu şirketine yönelik paylaşımları üzerinden duygu analizi yapılmış ve farklı modellerin performansları karşılaştırılmıştır.

- **Veri Kaynağı:** Kaggle'dan alınan *Twitter Airline Sentiment Dataset*. Veri seti, havayolu şirketleri hakkında pozitif, negatif ve nötr duygu içeren toplam 14.640 tweet içermektedir.
- **Amaç:** Sosyal medya verilerinden duygu analizi yaparak müşteri memnuniyetini analiz etmek ve sosyal medya analitiği için etkin yöntemler belirlemek.
- **Yöntem:** Çalışmada Makine Öğrenmesi (Naïve Bayes, Lojistik Regresyon, Rastgele Orman), Derin Öğrenme (Evrişimsel Sinir Ağları - CNN, Uzun Kısa Süreli Bellek - LSTM) ve Hibrit modeller kullanılarak karşılaştırmalı analizler yapılmıştır.
- **Dengesiz Veri Sorunu:** Veri seti dengesiz olduğu için, model performansını artırmak adına SMOTE (Sentetik Azınlık Aşırı Örnekleme Tekniği) yöntemi kullanılarak veri çoğaltma işlemi yapılmıştır.
- **En Yüksek Performans:** Rastgele Orman (Random Forest) algoritması, 0.88 F-Skor değeri ile en başarılı sonuçları elde etmiştir.

## Kullanılan Teknolojiler ve Kütüphaneler

-Python
-scikit-learn
-TensorFlow / PyTorch
-Pandas
-Numpy
-Matplotlib
-Searborn



