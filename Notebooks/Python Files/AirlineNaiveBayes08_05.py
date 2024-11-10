import tensorflow as tf
tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
tf.config.experimental_connect_to_cluster(tpu)
tf.tpu.experimental.initialize_tpu_system(tpu)
strategy = tf.distribute.TPUStrategy(tpu)

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
import numpy as np
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
nltk.download('punkt')
nltk.download('stopwords')
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from google.colab import drive

#Connecting to Google Drive
drive.mount('/content/drive')

#Loading the dataset from Google Colab
df = pd.read_csv('/content/drive/MyDrive/yuksekTez/airline_dataset/cleaned_text.csv')

# Create the feature matrix and target vector
X = df['clean_text1'].values
y = df['airline_sentiment'].values

df["clean_text1"] = df["clean_text1"].apply(str)


# Split the data into training and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

with strategy.scope():
  # Extract features from the preprocessed data using bag-of-words
  vectorizer = CountVectorizer()
  X_train = vectorizer.fit_transform(X_train)
  X_test = vectorizer.transform(X_test)

with strategy.scope():
  # Train the Naive Bayes classifier
  model = MultinomialNB()
  model.fit(X_train, y_train)

# Test the classifier on the test set
y_pred = model.predict(X_test)

# Evaluate the performance of the classifier
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
