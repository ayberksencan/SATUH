"""
Author: Ömer Ayberk ŞENCAN
Date: 11/10/2023
Description: This module implements logistic regression for sentiment analysis on airline tweets.
             It includes dataset preprocessing, hyperparameter tuning with GridSearchCV, and model evaluation.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from google.colab import drive

# Constants
FEATURES_PATH = '/content/drive/MyDrive/yuksekTez/airline_dataset/preprocessed_balanced_features.csv'
LABELS_PATH = '/content/drive/MyDrive/yuksekTez/airline_dataset/preprocessed_balanced_labels.csv'
TEST_SIZE = 0.25
RANDOM_STATE = 42
PARAM_GRID = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2']
}


def mount_drive():
    """Mounts Google Drive for data access."""
    drive.mount('/content/drive')


def load_data(features_path: str = FEATURES_PATH, labels_path: str = LABELS_PATH) -> tuple:
    """Loads preprocessed feature and label datasets."""
    features = pd.read_csv(features_path).values
    labels = pd.read_csv(labels_path)['airline_sentiment'].values
    return features, labels


def split_data(features, labels, test_size: float = TEST_SIZE, random_state: int = RANDOM_STATE) -> tuple:
    """Splits the dataset into training and testing sets."""
    return train_test_split(features, labels, test_size=test_size, random_state=random_state)


def train_logistic_regression(X_train, y_train) -> LogisticRegression:
    """Trains a logistic regression model."""
    model = LogisticRegression(solver='liblinear')
    model.fit(X_train, y_train)
    return model


def tune_hyperparameters(X_train, y_train) -> LogisticRegression:
    """Tunes hyperparameters using GridSearchCV and returns the best model."""
    grid_search = GridSearchCV(LogisticRegression(solver='liblinear'), PARAM_GRID, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    print("Best Hyperparameters:", grid_search.best_params_)
    return grid_search.best_estimator_


def evaluate_model(model, X_test, y_test):
    """Evaluates the model and prints performance metrics."""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))


def main():
    """Main function to execute the logistic regression sentiment analysis pipeline."""
    mount_drive()
    features, labels = load_data()
    X_train, X_test, y_train, y_test = split_data(features, labels)

    # Train logistic regression model
    print("Training Logistic Regression Model...")
    model = train_logistic_regression(X_train, y_train)
    evaluate_model(model, X_test, y_test)

    # Hyperparameter tuning
    print("\nTuning Hyperparameters...")
    best_model = tune_hyperparameters(X_train, y_train)
    evaluate_model(best_model, X_test, y_test)


if __name__ == "__main__":
    main()
