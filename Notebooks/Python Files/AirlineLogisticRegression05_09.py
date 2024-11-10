# Connect to TPU
tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
tf.config.experimental_connect_to_cluster(tpu)
tf.tpu.experimental.initialize_tpu_system(tpu)
strategy = tf.distribute.TPUStrategy(tpu)

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
from google.colab import drive
from sklearn.model_selection import train_test_split

# Connecting to Google Drive
drive.mount('/content/drive')

# Load the preprocessed and balanced features from CSV
df_resampled_features = pd.read_csv('/content/drive/MyDrive/yuksekTez/airline_dataset/preprocessed_balanced_features.csv')

# Load the balanced labels from CSV
df_resampled_labels = pd.read_csv('/content/drive/MyDrive/yuksekTez/airline_dataset/preprocessed_balanced_labels.csv')

with strategy.scope():
    # Create the feature matrix (X) and target vector (y)
    X = df_resampled_features.values
    y = df_resampled_labels['airline_sentiment'].values

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # Train a logistic regression model
    model = LogisticRegression(solver='liblinear')
    model.fit(X_train, y_train)

    # Test the model on the test set
    y_pred = model.predict(X_test)

# Evaluate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)

# Calculate and print additional metrics
print('Classification Report:\n', classification_report(y_test, y_pred))
print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred))

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV

# Load the preprocessed and balanced dataset
# Load the preprocessed and balanced dataset
X = df_resampled_features.values
y = df_resampled_labels['airline_sentiment'].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Define a hyperparameter grid for tuning
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],  # Adjust as needed
    'penalty': ['l1', 'l2']  # Adjust as needed
}

# Perform hyperparameter tuning using GridSearchCV
grid_search = GridSearchCV(LogisticRegression(solver='liblinear'), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_
best_lr_model = grid_search.best_estimator_

# Train the best Logistic Regression model
best_lr_model.fit(X_train, y_train)
y_pred = best_lr_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Print the best hyperparameters found by GridSearchCV
print("Best Hyperparameters:", best_params)

# Print classification report and confusion matrix
print(classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
