# SATUH
SATUH (Sentiment Analysis on Twitter Using Machine Learning, Deep Learning, and Hybrid Models)

---

## EN

### Twitter Sentiment Analysis

This project is a master’s thesis conducted by Ömer Ayberk ŞENCAN at Gazi University, Graduate School of Natural and Applied Sciences, in the Department of Computer Engineering. It aims to analyze sentiment on Twitter data related to major airline companies using machine learning, deep learning, and hybrid models.

#### Project Summary
This study utilizes artificial intelligence-based models to analyze data generated on social media platforms. Sentiment analysis was conducted on Twitter posts related to six major airline companies in the USA, and the performance of various models was compared.

- **Data Source:** The *Twitter Airline Sentiment Dataset* from Kaggle, which contains a total of 14,640 tweets labeled as positive, negative, or neutral.
- **Goal:** To analyze customer satisfaction based on social media sentiment and to identify effective methods for social media analytics.
- **Methodology:** Machine Learning (Naïve Bayes, Logistic Regression, Random Forest), Deep Learning (Convolutional Neural Networks - CNN, Long Short-Term Memory - LSTM), and Hybrid models were used for comparative analysis.
- **Data Imbalance Issue:** Due to the imbalanced nature of the dataset, SMOTE (Synthetic Minority Over-sampling Technique) was applied to balance the data.
- **Best Performance:** The Random Forest algorithm achieved the highest performance with an F-Score of 0.88.

#### Technologies and Libraries Used

Various libraries and platforms were utilized to perform sentiment analysis using machine learning and deep learning. The primary tools and technologies used are as follows:

- **Python**: The core programming language for the project.
- **Google Colaboratory**: An online development environment provided by Google with free access to GPUs. Popular libraries like Keras, TensorFlow, and PyTorch are also available on this platform.
- **Machine Learning Algorithms**:
  - **Naïve Bayes**: A commonly used algorithm for text classification.
  - **Random Forest**: A decision tree-based method that improves accuracy by using multiple trees.
  - **Logistic Regression**: A widely used algorithm for binary classification problems.
  - **Support Vector Classifier (SVC)**: Provides strong results for classification and regression analyses.
- **Deep Learning Models**:
  - **Convolutional Neural Network (CNN)**: Used for feature extraction and classification in image and text data.
  - **Long Short-Term Memory (LSTM)**: A network structure capable of learning long-term dependencies, commonly used for time series data analysis.
- **Hybrid Models**:
  - **CNN-Gated Recurrent Unit (GRU)**: A hybrid structure combining CNN and GRU optimized for sentiment analysis.
  - **LSTM-GRU Based Models**: Hybrid models created by combining LSTM and GRU, achieving strong results in text sentiment analysis.
- **SMOTE (Synthetic Minority Over-sampling Technique)**: Used to correct dataset imbalance, aiming to balance class distribution.
- **Natural Language Processing (NLP) Techniques**:
  - **TF-IDF (Term Frequency-Inverse Document Frequency)**: A method used for text feature extraction by evaluating the importance of each term.
- **Other Python Libraries**:
  - **Pandas and NumPy**: For data processing and feature extraction.
  - **Matplotlib and Seaborn**: For data visualization.
  - **scikit-learn**: For model training, data splitting, and performance evaluation metrics.
  - **TensorFlow / PyTorch**: For building and training deep learning models.

---

## TR

### Twitter Duygu Analizi

Bu proje, Ömer Ayberk ŞENCAN tarafından Gazi Üniversitesi Fen Bilimleri Enstitüsü Bilgisayar Mühendisliği Bölümü için hazırlanmış bir yüksek lisans tez çalışmasıdır. Çalışma, büyük havayolu şirketleriyle ilgili Twitter verileri üzerinden duygu analizi yapmayı amaçlar.

#### Proje Özeti
Bu çalışmada, sosyal medya platformlarında üretilen verilerin yapay zekâ temelli modeller ile analiz edilmesi amaçlanmıştır. Twitter kullanıcılarının ABD'deki altı büyük havayolu şirketine yönelik paylaşımları üzerinden duygu analizi yapılmış ve farklı modellerin performansları karşılaştırılmıştır.

- **Veri Kaynağı:** Kaggle'dan alınan *Twitter Airline Sentiment Dataset*. Veri seti, havayolu şirketleri hakkında pozitif, negatif ve nötr duygu içeren toplam 14.640 tweet içermektedir.
- **Amaç:** Sosyal medya verilerinden duygu analizi yaparak müşteri memnuniyetini analiz etmek ve sosyal medya analitiği için etkin yöntemler belirlemek.
- **Yöntem:** Çalışmada Makine Öğrenmesi (Naïve Bayes, Lojistik Regresyon, Rastgele Orman), Derin Öğrenme (Evrişimsel Sinir Ağları - CNN, Uzun Kısa Süreli Bellek - LSTM) ve Hibrit modeller kullanılarak karşılaştırmalı analizler yapılmıştır.
- **Dengesiz Veri Sorunu:** Veri seti dengesiz olduğu için, model performansını artırmak adına SMOTE (Sentetik Azınlık Aşırı Örnekleme Tekniği) yöntemi kullanılarak veri çoğaltma işlemi yapılmıştır.
- **En Yüksek Performans:** Rastgele Orman (Random Forest) algoritması, 0.88 F-Skor değeri ile en başarılı sonuçları elde etmiştir.

#### Kullanılan Teknolojiler ve Kütüphaneler

Bu proje kapsamında, makine öğrenmesi ve derin öğrenme tabanlı duygu analizi gerçekleştirmek için çeşitli kütüphaneler ve platformlar kullanılmıştır. Aşağıda kullanılan ana araçlar ve teknolojiler listelenmiştir:

- **Python**: Projede kullanılan temel programlama dili.
- **Google Colaboratory**: Google tarafından sağlanan ve ücretsiz GPU erişimi sunan bir çevrimiçi geliştirme ortamı. Bu platformda Keras, TensorFlow ve PyTorch gibi popüler kütüphaneler de ücretsiz olarak kullanılabilmektedir.
- **Makine Öğrenmesi Algoritmaları**:
  - **Naïve Bayes**: Metin sınıflandırma için kullanılan temel algoritmalardan biridir.
  - **Rastgele Orman (Random Forest)**: Karar ağacı bazlı, çok sayıda ağaç kullanarak doğruluğu artıran bir yöntem.
  - **Lojistik Regresyon (Logistic Regression)**: İkili sınıflandırma problemlerinde yaygın olarak kullanılan bir algoritma.
  - **Destek Vektör Sınıflandırıcı (Support Vector Classifier)**: Sınıflandırma ve regresyon analizinde güçlü sonuçlar sağlayan makine öğrenmesi algoritması.
- **Derin Öğrenme Modelleri**:
  - **Evrişimsel Sinir Ağı (CNN)**: Görüntü ve metin verilerinde öznitelik çıkarımı ve sınıflandırma işlemleri için kullanılan derin öğrenme modeli.
  - **Uzun-Kısa Süreli Bellek (LSTM)**: Zaman serisi verilerinin analizinde kullanılan, bellek hücreleri ile uzun vadeli bağımlılıkları öğrenebilen bir ağ yapısı.
- **Hibrit Modeller**:
  - **CNN-Gated Recurrent Unit (GRU)**: CNN ile GRU'nun birleştirildiği hibrit bir yapı, duygu analizi için optimize edilmiştir.
  - **LSTM-GRU Tabanlı Modeller**: LSTM ve GRU yapılarının kombinasyonları kullanılarak oluşturulan, metin duygu analizinde güçlü sonuçlar elde eden hibrit modeller.
- **SMOTE (Synthetic Minority Over-sampling Technique)**: Veri setinin dengesiz yapısını düzeltmek için kullanılan bir çoğaltma tekniği. Bu proje kapsamında dengesiz sınıf verilerini dengelemek amacıyla kullanılmıştır.
- **Doğal Dil İşleme (NLP) Teknikleri**:
  - **TF-IDF (Term Frequency-Inverse Document Frequency)**: Metin öznitelik çıkarımı için kullanılan bir yöntemdir. Her bir terimin önemini değerlendirerek metinleri sayısal verilere dönüştürmektedir.
- **Diğer Python Kütüphaneleri**:
  - **Pandas ve NumPy**: Veri işleme ve öznitelik çıkarımı.
  - **Matplotlib ve Seaborn**: Veri görselleştirme.
  - **scikit-learn**: Model eğitimi, veri bölme ve performans değerlendirme metrikleri.
  - **TensorFlow / PyTorch**: Derin öğrenme modellerinin kurulması ve eğitilmesi için.
