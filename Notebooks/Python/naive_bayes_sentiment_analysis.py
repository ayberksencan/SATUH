"""
Author: Ömer Ayberk ŞENCAN
Date: 08/05/2023
Description: This module implements a Naive Bayes classifier for sentiment analysis on airline tweets.
             It includes text preprocessing, feature extraction, training, and evaluation.
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from google.colab import drive
import nltk

# Constants
DATASET_PATH = '/content/drive/MyDrive/yuksekTez/airline_dataset/cleaned_text.csv'
TEST_SIZE = 0.2
RANDOM_STATE = 42


def mount_drive():
    """Mounts Google Drive for data access."""
    drive.mount('/content/drive')


def load_data(path: str = DATASET_PATH) -> pd.DataFrame:
    """Loads the dataset from a CSV file."""
    return pd.read_csv(path)


def preprocess_text(df: pd.DataFrame, text_column: str = "clean_text1") -> pd.Series:
    """Applies basic text preprocessing to a text column."""
    nltk.download('punkt')
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

    def clean_text(text):
        text = text.lower()
        text = re.sub(f"[{string.punctuation}]", "", text)  # Remove punctuation
        tokens = word_tokenize(text)
        tokens = [word for word in tokens if word not in stop_words]
        return " ".join(tokens)

    df[text_column] = df[text_column].astype(str).apply(clean_text)
    return df[text_column]


def split_data(features, labels, test_size=TEST_SIZE, random_state=RANDOM_STATE) -> tuple:
    """Splits the dataset into training and testing sets."""
    return train_test_split(features, labels, test_size=test_size, random_state=random_state)


def extract_features(train_data, test_data) -> tuple:
    """Converts text data into numerical features using bag-of-words."""
    vectorizer = CountVectorizer()
    X_train = vectorizer.fit_transform(train_data)
    X_test = vectorizer.transform(test_data)
    return X_train, X_test, vectorizer


def train_naive_bayes(X_train, y_train) -> MultinomialNB:
    """Trains a Naive Bayes classifier."""
    model = MultinomialNB()
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """Evaluates the model and prints performance metrics."""
    y_pred = model.predict(X_test)
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))


def main():
    """Main function to execute the Naive Bayes sentiment analysis pipeline."""
    mount_drive()
    df = load_data()

    # Preprocess text data
    df["clean_text1"] = preprocess_text(df)

    # Create feature and label matrices
    X = df["clean_text1"].values
    y = df["airline_sentiment"].values

    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Extract features
    X_train_features, X_test_features, vectorizer = extract_features(X_train, X_test)

    # Train and evaluate the model
    model = train_naive_bayes(X_train_features, y_train)
    evaluate_model(model, X_test_features, y_test)


if __name__ == "__main__":
    main()
