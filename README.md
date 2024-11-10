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

Bu proje kapsamında, makine öğrenmesi ve derin öğrenme tabanlı duygu analizi gerçekleştirmek için çeşitli kütüphaneler ve platformlar kullanılmıştır. Aşağıda kullanılan ana araçlar ve teknolojiler listelenmiştir:

- **Python**: Projede kullanılan temel programlama dili.
- **Google Colaboratory**: Google tarafından sağlanan ve ücretsiz GPU erişimi sunan bir çevrimiçi geliştirme ortamı. Bu platformda Keras, TensorFlow ve PyTorch gibi popüler kütüphaneler de ücretsiz olarak kullanılabilmektedir&#8203;:contentReference[oaicite:0]{index=0}.
- **Makine Öğrenmesi Algoritmaları**:
  - **Naïve Bayes**: Metin sınıflandırma için kullanılan temel algoritmalardan biridir.
  - **Rastgele Orman (Random Forest)**: Karar ağacı bazlı, çok sayıda ağaç kullanarak doğruluğu artıran bir yöntem.
  - **Lojistik Regresyon (Logistic Regression)**: İkili sınıflandırma problemlerinde yaygın olarak kullanılan bir algoritma.
  - **Destek Vektör Sınıflandırıcı (Support Vector Classifier)**: Sınıflandırma ve regresyon analizinde güçlü sonuçlar sağlayan makine öğrenmesi algoritması&#8203;:contentReference[oaicite:1]{index=1}.
- **Derin Öğrenme Modelleri**:
  - **Evrişimsel Sinir Ağı (CNN)**: Görüntü ve metin verilerinde öznitelik çıkarımı ve sınıflandırma işlemleri için kullanılan derin öğrenme modeli.
  - **Uzun-Kısa Süreli Bellek (LSTM)**: Zaman serisi verilerinin analizinde kullanılan, bellek hücreleri ile uzun vadeli bağımlılıkları öğrenebilen bir ağ yapısı&#8203;:contentReference[oaicite:2]{index=2}.
- **Hibrit Modeller**:
  - **CNN-Gated Recurrent Unit (GRU)**: CNN ile GRU'nun birleştirildiği hibrit bir yapı, duygu analizi için optimize edilmiştir.
  - **LSTM-GRU Tabanlı Modeller**: LSTM ve GRU yapılarının kombinasyonları kullanılarak oluşturulan, metin duygu analizinde güçlü sonuçlar elde eden hibrit modeller&#8203;:contentReference[oaicite:3]{index=3}.
- **SMOTE (Synthetic Minority Over-sampling Technique)**: Veri setinin dengesiz yapısını düzeltmek için kullanılan bir çoğaltma tekniği. Bu proje kapsamında dengesiz sınıf verilerini dengelemek amacıyla kullanılmıştır&#8203;:contentReference[oaicite:4]{index=4}.
- **Doğal Dil İşleme (NLP) Teknikleri**:
  - **TF-IDF (Term Frequency-Inverse Document Frequency)**: Metin öznitelik çıkarımı için kullanılan bir yöntemdir. Her bir terimin önemini değerlendirerek metinleri sayısal verilere dönüştürmektedir&#8203;:contentReference[oaicite:5]{index=5}.
- **Diğer Python Kütüphaneleri**:
  - **Pandas ve NumPy**: Veri işleme ve öznitelik çıkarımı.
  - **Matplotlib ve Seaborn**: Veri görselleştirme.
  - **scikit-learn**: Model eğitimi, veri bölme ve performans değerlendirme metrikleri.
  - **TensorFlow / PyTorch**: Derin öğrenme modellerinin kurulması ve eğitilmesi için.

