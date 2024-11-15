"""
Author: Ömer Ayberk ŞENCAN
Date: 04/09/2023
Description: This module implements a Random Forest classifier for sentiment analysis on airline tweets.
             It includes loading preprocessed data, training the model, and evaluating performance metrics.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from google.colab import drive

# Constants
FEATURES_PATH = '/content/drive/MyDrive/yuksekTez/airline_dataset/preprocessed_balanced_features.csv'
LABELS_PATH = '/content/drive/MyDrive/yuksekTez/airline_dataset/preprocessed_balanced_labels.csv'
TEST_SIZE = 0.25
RANDOM_STATE = 42
N_ESTIMATORS = 500


def mount_drive():
    """Mounts Google Drive for data access."""
    drive.mount('/content/drive')


def load_data(features_path: str = FEATURES_PATH, labels_path: str = LABELS_PATH) -> tuple:
    """Loads preprocessed feature and label datasets."""
    features = pd.read_csv(features_path).values
    labels = pd.read_csv(labels_path)['airline_sentiment'].values
    return features, labels


def split_data(features, labels, test_size=TEST_SIZE, random_state=RANDOM_STATE) -> tuple:
    """Splits the dataset into training and testing sets."""
    return train_test_split(features, labels, test_size=test_size, random_state=random_state)


def train_random_forest(X_train, y_train, n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE) -> RandomForestClassifier:
    """Trains a Random Forest classifier."""
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """Evaluates the Random Forest model and prints performance metrics."""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))


def save_predictions(y_pred, y_test, path: str):
    """Saves predictions to a CSV file."""
    df_pred = pd.DataFrame({"predicted_values": y_pred, "real_values": y_test})
    df_pred.to_csv(path, index=False)
    print(f"Predictions saved to {path}")


def main():
    """Main function to execute the Random Forest sentiment analysis pipeline."""
    mount_drive()
    features, labels = load_data()
    X_train, X_test, y_train, y_test = split_data(features, labels)

    # Train Random Forest model
    print("Training Random Forest Model...")
    model = train_random_forest(X_train, y_train)

    # Evaluate the model
    print("\nEvaluating the Model...")
    evaluate_model(model, X_test, y_test)


if __name__ == "__main__":
    main()
