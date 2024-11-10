import tensorflow as tf
import os
# Connect to TPU
resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='grpc://' + os.environ['COLAB_TPU_ADDR'])
tf.config.experimental_connect_to_cluster(resolver)
tf.tpu.experimental.initialize_tpu_system(resolver)
strategy = tf.distribute.TPUStrategy(resolver)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
from google.colab import drive

# Mount Google Drive
drive.mount('/content/drive')

# Define the path to the preprocessed dataset
preprocessed_data_path = '/content/drive/MyDrive/yuksekTez/airline_dataset/preprocessed_data18-09.csv'

# Load the preprocessed data
df = pd.read_csv(preprocessed_data_path)

# Prepare data
X = df.drop('label', axis=1).values
y = df['label'].values

# Reshape the input data to be 3D
X = X.reshape(X.shape[0], X.shape[1], 1)

# Split data into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.25, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.1, random_state=42)

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

# Reshape the input data to be 2D
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1])
X_val = X_val.reshape(X_val.shape[0], X_val.shape[1])
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1])

# Initialize a base learner (Decision Tree with max depth 1)
base_learner = DecisionTreeClassifier(max_depth=1)

# Initialize the AdaBoost classifier
adaboost_classifier = AdaBoostClassifier(base_estimator=base_learner, n_estimators=50, random_state=42)

# Initialize history
history = {'train_accuracy': [], 'test_accuracy': []}

for i in range(adaboost_classifier.n_estimators):
    # Train the AdaBoost classifier
    adaboost_classifier.fit(X_train, y_train)

    # Evaluate the model on training data
    train_accuracy = adaboost_classifier.score(X_train, y_train)

    # Evaluate the model on test data
    test_accuracy = adaboost_classifier.score(X_test, y_test)

    # Append accuracies to history
    history['train_accuracy'].append(train_accuracy)
    history['test_accuracy'].append(test_accuracy)

    # Print progress
    print(f"Iteration {i+1}/{adaboost_classifier.n_estimators} - Train Accuracy: {train_accuracy:.4f} - Test Accuracy: {test_accuracy:.4f}")

# Generate predictions
y_pred = adaboost_classifier.predict(X_test)

# Print classification report
print(classification_report(y_test, y_pred))


# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test accuracy: {accuracy * 100:.2f}%')

# Generate predictions
y_pred = np.argmax(model.predict(X_test), axis=-1)

# Print classification report
print(classification_report(y_test, y_pred))

import matplotlib.pyplot as plt

# Access training history
training_loss = history.history['loss']
validation_loss = history.history['val_loss']
training_accuracy = history.history['accuracy']
validation_accuracy = history.history['val_accuracy']

# Plot training and validation loss
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(training_loss, label='Training Loss')
plt.plot(validation_loss, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Plot training and validation accuracy
plt.subplot(1, 2, 2)
plt.plot(training_accuracy, label='Training Accuracy')
plt.plot(validation_accuracy, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

