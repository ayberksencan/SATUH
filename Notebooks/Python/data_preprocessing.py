"""
Author: Ömer Ayberk ŞENCAN
Date: 04/05/2023
Description: This module preprocesses airline tweets for sentiment analysis. It includes cleaning,
             tokenization, balancing with SMOTE, and saving the preprocessed dataset.
"""

import pandas as pd
import numpy as np
import re
import string
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from google.colab import drive

# Constants
DATASET_PATH = '/content/drive/MyDrive/yuksekTez/airline_dataset/Tweets.csv'
CLEANED_TEXT_PATH = '/content/drive/MyDrive/yuksekTez/airline_dataset/cleaned_text1.csv'
FEATURES_PATH = '/content/drive/MyDrive/yuksekTez/airline_dataset/preprocessed_balanced_features.csv'
LABELS_PATH = '/content/drive/MyDrive/yuksekTez/airline_dataset/preprocessed_balanced_labels.csv'
BALANCED_DATASET_PATH = '/content/drive/MyDrive/yuksekTez/airline_dataset/preprocessed_balanced_dataset.csv'
VOCAB_SIZE = 10000
MAXLEN = 100
RANDOM_STATE = 42


def mount_drive():
    """Mounts Google Drive for data access."""
    drive.mount('/content/drive')


def load_data(path: str = DATASET_PATH) -> pd.DataFrame:
    """Loads the dataset from a CSV file."""
    return pd.read_csv(path)


def clean_text(text: str) -> str:
    """Cleans the text by removing URLs, mentions, hashtags, punctuation, numbers, and stopwords."""
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'@\w+', '', text)  # Remove mentions
    text = re.sub(r'#(\w+)', r'\1', text)  # Remove hashtags but keep text
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = ' '.join(word for word in text.split() if word.lower() not in stop_words)  # Remove stopwords
    return text.lower()


def preprocess_text_column(df: pd.DataFrame, text_column: str = "text") -> pd.DataFrame:
    """Applies text cleaning to a specified column."""
    df['clean_text1'] = df[text_column].astype(str).apply(clean_text)
    return df


def check_data_distribution(df: pd.DataFrame, label_column: str = "airline_sentiment"):
    """Prints class distributions and checks for NaN values."""
    print(f"Unique values in {label_column}:", df[label_column].unique())
    print(f"Value counts in {label_column}:\n", df[label_column].value_counts())
    print(f"Number of NaNs in cleaned text:", df['clean_text1'].isna().sum())


def tokenize_and_pad_sequences(df: pd.DataFrame, text_column: str = "clean_text1") -> tuple:
    """Tokenizes and pads text sequences."""
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
    tokenizer.fit_on_texts(df[text_column])
    sequences = tokenizer.texts_to_sequences(df[text_column])
    padded_sequences = pad_sequences(sequences, maxlen=MAXLEN, padding="post", truncating="post")
    vocab_size = len(tokenizer.word_index) + 1
    print("Maximum Sequence Length:", MAXLEN)
    print("Vocab Size:", vocab_size)
    return padded_sequences, tokenizer


def balance_data_with_smote(X, y) -> tuple:
    """Applies SMOTE to balance the dataset."""
    smote = SMOTE(random_state=RANDOM_STATE)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    return X_resampled, y_resampled


def save_to_csv(df: pd.DataFrame, path: str):
    """Saves a DataFrame to a CSV file."""
    df.to_csv(path, index=False)
    print(f"Data saved to {path}")


def main():
    """Main function to execute preprocessing pipeline."""
    mount_drive()
    df = load_data()

    # Preprocess text
    df = preprocess_text_column(df)

    # Select relevant columns
    selected_columns = ["text", "airline_sentiment", "clean_text1"]
    df_selected = df[selected_columns].dropna(subset=['clean_text1']).reset_index(drop=True)

    # Check data distribution
    check_data_distribution(df_selected)

    # Vectorize text with TF-IDF
    tfidf_vectorizer = TfidfVectorizer()
    X_tfidf = tfidf_vectorizer.fit_transform(df_selected['clean_text1'])

    # Balance dataset with SMOTE
    X_resampled, y_resampled = balance_data_with_smote(X_tfidf, df_selected['airline_sentiment'].values)

    # Save preprocessed data
    save_to_csv(pd.DataFrame(X_resampled.toarray(), columns=tfidf_vectorizer.get_feature_names_out()), FEATURES_PATH)
    save_to_csv(pd.DataFrame({'airline_sentiment': y_resampled}), LABELS_PATH)
    save_to_csv(df_selected, CLEANED_TEXT_PATH)

    # Combine features and labels for convenience
    df_resampled = pd.DataFrame(data=X_resampled.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
    df_resampled['airline_sentiment'] = y_resampled
    save_to_csv(df_resampled, BALANCED_DATASET_PATH)


if __name__ == "__main__":
    main()
