"""
Author: Ömer Ayberk ŞENCAN
Date: 04/09/2023
Description: This module implements a Naive Bayes classifier with cross-validation for sentiment
             analysis on airline tweets. It includes training, evaluation, and cross-validation.
"""

import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from google.colab import drive

# Constants
FEATURES_PATH = '/content/drive/MyDrive/yuksekTez/airline_dataset/preprocessed_balanced_features.csv'
LABELS_PATH = '/content/drive/MyDrive/yuksekTez/airline_dataset/preprocessed_balanced_labels.csv'
TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_FOLDS = 5


def mount_drive():
    """Mounts Google Drive for data access."""
    drive.mount('/content/drive')


def load_data(features_path: str = FEATURES_PATH, labels_path: str = LABELS_PATH) -> tuple:
    """Loads preprocessed features and labels from CSV files."""
    features = pd.read_csv(features_path).values
    labels = pd.read_csv(labels_path)['airline_sentiment'].values
    return features, labels


def split_data(features, labels, test_size=TEST_SIZE, random_state=RANDOM_STATE) -> tuple:
    """Splits the dataset into training and testing sets."""
    return train_test_split(features, labels, test_size=test_size, random_state=random_state)


def perform_cross_validation(model, X_train, y_train, cv_folds=CV_FOLDS) -> float:
    """Performs cross-validation and returns the mean score."""
    scores = cross_val_score(model, X_train, y_train, cv=cv_folds)
    print("Cross-Validation Scores:", scores)
    mean_score = np.mean(scores)
    print("Mean CV Score:", mean_score)
    return mean_score


def train_model(model, X_train, y_train):
    """Trains the Naive Bayes model on the training data."""
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
    features, labels = load_data()
    X_train, X_test, y_train, y_test = split_data(features, labels)

    # Initialize Naive Bayes model
    model = MultinomialNB()

    # Perform cross-validation
    print("Performing Cross-Validation...")
    perform_cross_validation(model, X_train, y_train)

    # Train the model on the full training set
    print("\nTraining the Naive Bayes Model...")
    trained_model = train_model(model, X_train, y_train)

    # Evaluate the model
    print("\nEvaluating the Model...")
    evaluate_model(trained_model, X_test, y_test)


if __name__ == "__main__":
    main()
