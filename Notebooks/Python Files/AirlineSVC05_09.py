import tensorflow as tf
# Connect to TPU
tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
tf.config.experimental_connect_to_cluster(tpu)
tf.tpu.experimental.initialize_tpu_system(tpu)
strategy = tf.distribute.TPUStrategy(tpu)

from google.colab import drive
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

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

    # Train a Support Vector Classifier (SVC)
    model = SVC()
    model.fit(X_train, y_train)

    # Test the model on the test set
    y_pred = model.predict(X_test)

# Evaluate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)

# Calculate and print additional metrics
print('Classification Report:\n', classification_report(y_test, y_pred))
print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred))

# Import necessary libraries
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from google.colab import drive

# Connecting to Google Drive
drive.mount('/content/drive')

# Load the preprocessed and balanced dataset
df = pd.read_csv('/content/drive/MyDrive/yuksekTez/airline_dataset/preprocessed_balanced_features.csv')

# Create the feature matrix (X) and target vector (y)
X = df_resampled_features.values
y = df_resampled_labels['airline_sentiment'].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Create a Support Vector Classifier
with strategy.scope():
    svc = SVC()  # You can specify kernel and other hyperparameters here

# Define a hyperparameter grid for tuning
param_grid = {
    'C': [0.1, 1, 10],  # Regularization parameter
    'kernel': ['linear', 'rbf', 'poly'],  # Kernel types to try
    'gamma': ['scale', 'auto', 1, 0.1],  # Kernel coefficient
}

# Use GridSearchCV for hyperparameter tuning
grid_search = GridSearchCV(estimator=svc, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)

# Fit the grid search to the data
grid_search.fit(X_train, y_train)

# Get the best hyperparameters from the grid search
best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)

# Create a new SVC model with the best hyperparameters
with strategy.scope():
    best_svc = SVC(**best_params)

# Train the model on the training data
with strategy.scope():
    best_svc.fit(X_train, y_train)

# Test the classifier on the test set
with strategy.scope():
    y_pred = best_svc.predict(X_test)

# Evaluate the performance of the classifier
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
