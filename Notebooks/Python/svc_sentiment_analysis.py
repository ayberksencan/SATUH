"""
Author: Ömer Ayberk ŞENCAN
Date: 05/09/2023
Description: This module implements a Support Vector Classifier (SVC) for sentiment analysis on airline tweets.
             It includes model training, hyperparameter tuning with GridSearchCV, and evaluation.
"""

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from google.colab import drive

# Constants
FEATURES_PATH = '/content/drive/MyDrive/yuksekTez/airline_dataset/preprocessed_balanced_features.csv'
LABELS_PATH = '/content/drive/MyDrive/yuksekTez/airline_dataset/preprocessed_balanced_labels.csv'
TEST_SIZE = 0.25
RANDOM_STATE = 42
PARAM_GRID = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto', 1, 0.1],
}


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


def train_svc(X_train, y_train) -> SVC:
    """Trains a basic Support Vector Classifier."""
    model = SVC()
    model.fit(X_train, y_train)
    return model


def tune_hyperparameters(X_train, y_train, param_grid=PARAM_GRID) -> SVC:
    """Tunes hyperparameters using GridSearchCV and returns the best model."""
    grid_search = GridSearchCV(SVC(), param_grid, cv=5, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
    print("Best Hyperparameters:", grid_search.best_params_)
    return grid_search.best_estimator_


def evaluate_model(model, X_test, y_test):
    """Evaluates the model and prints performance metrics."""
    y_pred = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))


def main():
    """Main function to execute the SVC sentiment analysis pipeline."""
    mount_drive()
    features, labels = load_data()
    X_train, X_test, y_train, y_test = split_data(features, labels)

    # Train initial SVC model
    print("Training Initial SVC Model...")
    initial_model = train_svc(X_train, y_train)
    evaluate_model(initial_model, X_test, y_test)

    # Hyperparameter tuning
    print("\nTuning Hyperparameters...")
    best_model = tune_hyperparameters(X_train, y_train)

    # Evaluate the best model
    print("\nEvaluating the Best Model...")
    evaluate_model(best_model, X_test, y_test)


if __name__ == "__main__":
    main()
