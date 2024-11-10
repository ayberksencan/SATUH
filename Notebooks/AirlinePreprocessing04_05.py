#Importing and starting the TPU
import tensorflow as tf
tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
tf.config.experimental_connect_to_cluster(tpu)
tf.tpu.experimental.initialize_tpu_system(tpu)
strategy = tf.distribute.TPUStrategy(tpu)

#!pip install -U imbalanced-learn

#Importing necessary libraries
import pandas as pd
import numpy as np
import re
import string
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from imblearn.over_sampling import SMOTE  # Import SMOTE
from sklearn.feature_extraction.text import TfidfVectorizer

#Connecting to Google Drive to get the dataset
from google.colab import drive
drive.mount('/content/drive')

#Loading the dataset from Google Drive
df = pd.read_csv('/content/drive/MyDrive/yuksekTez/airline_dataset/Tweets.csv')

#Cleaining the text data
def clean_text(text):
  #Removing the URL's
  text = re.sub(r'http\S+', '', text)
  # remove mentions
  text = re.sub(r'@\w+', '', text)
  # remove hashtags (Only the Hastag sign not the text itself)
  processed_text = re.sub(r'#(\w+)', r'\1', text)
  # remove punctuation
  text = text.translate(str.maketrans('', '', string.punctuation))
  # remove numbers
  text = re.sub(r'\d+', '', text)
  # remove stopwords
  text = ' '.join([word for word in text.split() if word.lower() not in stopwords.words('english')])
  return text.lower()
  df.head()

#Checking the tail of the dataset

#Appying clean_text function to the dataset
with strategy.scope():
  df['clean_text1'] = df['text'].apply(lambda x: clean_text(x))

df.head()

# Select and isolate some columns
selected_columns = ["text", "airline_sentiment", "clean_text1"]
df_selected = df[selected_columns]

# Print the selected dataframe
print(df_selected)

df_selected.tail()

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
# Tokenize and pad sequences within the strategy scope
with strategy.scope():
    tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
    tokenizer.fit_on_texts(df['clean_text1'])
    vocab_size = len(tokenizer.word_index) + 1
    sequences = tokenizer.texts_to_sequences(df['clean_text1'])
    maxlen = 100
    padded_sequences = pad_sequences(sequences, maxlen=maxlen, padding="post", truncating="post")
    print("Maximum Sequence Length:", maxlen)
    print("Vocab Size: ", vocab_size)

# Check the number of items in a column
num_items = len(df_selected["clean_text1"])
print("Number of items in clean_text1:", num_items)

# Check the separation of item values in a column
unique_values = df_selected["airline_sentiment"].unique()
print("Unique values in airline_sentiment:", unique_values)

# Get the number of NaNs in a column
num_nans = df_selected["clean_text1"].isna().sum()
print(num_nans)

item_counts = df_selected['airline_sentiment'].value_counts()
print(item_counts)

# Drop rows with NaN values in the 'clean_text1' column
df_selected = df_selected.dropna(subset=['clean_text1'])

# Reset the index after removing rows
df_selected.reset_index(drop=True, inplace=True)

# Check if NaN rows are removed
print("Number of items after removing NaN rows:", len(df_selected))


# Vectorize the text data using TF-IDF
tfidf_vectorizer = TfidfVectorizer()
X_tfidf = tfidf_vectorizer.fit_transform(df['clean_text1'])

# Apply SMOTE to balance the dataset
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_tfidf, df['airline_sentiment'].values)

# Save the preprocessed and balanced features (X_resampled) to a CSV file
df_resampled_features = pd.DataFrame(data=X_resampled.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
df_resampled_features.to_csv('/content/drive/MyDrive/yuksekTez/airline_dataset/preprocessed_balanced_features.csv', index=False)

# Save the balanced labels (y_resampled) to a CSV file
df_resampled_labels = pd.DataFrame({'airline_sentiment': y_resampled})
df_resampled_labels.to_csv('/content/drive/MyDrive/yuksekTez/airline_dataset/preprocessed_balanced_labels.csv', index=False)

item_counts = df_resampled_labels['airline_sentiment'].value_counts()
print(item_counts)

#Saving the cleaned text column to a CSV file
df_selected.to_csv('/content/drive/MyDrive/yuksekTez/airline_dataset/cleaned_text1.csv', index=False)

# Create a DataFrame with both features and labels
columns = tfidf_vectorizer.get_feature_names_out()
df_resampled = pd.DataFrame(data=X_resampled.toarray(), columns=columns)
df_resampled['airline_sentiment'] = y_resampled

# Save the combined dataset to a CSV file
df_resampled.to_csv('/content/drive/MyDrive/yuksekTez/airline_dataset/preprocessed_balanced_dataset.csv', index=False)
