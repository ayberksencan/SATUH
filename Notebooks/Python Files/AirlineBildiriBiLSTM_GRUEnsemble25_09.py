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
from tensorflow.keras.layers import GRU, Dense, Bidirectional
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

from tensorflow.keras.callbacks import EarlyStopping

# Define the model-building function for BiLSTM
def build_bilstm_model():
    model = Sequential([
        Bidirectional(GRU(64, input_shape=(X.shape[1], X.shape[2]))),  # BiLSTM
        Dense(3, activation='softmax')  # 3 output neurons for 3 classes, softmax for multi-class classification
    ])
    return model

# Define the model-building function for GRU
def build_gru_model():
    model = Sequential([
        GRU(64, input_shape=(X.shape[1], X.shape[2])),
        Dense(3, activation='softmax')  # 3 output neurons for 3 classes, softmax for multi-class classification
    ])
    return model



# Use the strategy scope to distribute the training for BiLSTM
with strategy.scope():
    bilstm_model = build_bilstm_model()
    bilstm_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Define EarlyStopping callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

    # Train the BiLSTM model with early stopping
    history_bilstm = bilstm_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=64, callbacks=[early_stopping])



# Use the strategy scope to distribute the training for GRU
with strategy.scope():
    gru_model = build_gru_model()
    gru_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Define EarlyStopping callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

    # Train the GRU model with early stopping
    history_gru = gru_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=64, callbacks=[early_stopping])



# Generate predictions for BiLSTM and GRU
bilstm_predictions = np.argmax(bilstm_model.predict(X_test), axis=-1)
gru_predictions = np.argmax(gru_model.predict(X_test), axis=-1)



# Ensemble using a simple voting mechanism
ensemble_predictions = np.round((bilstm_predictions + gru_predictions) / 2).astype(int)

# Evaluate ensemble model
ensemble_accuracy = np.mean(ensemble_predictions == y_test)
print(f'Ensemble Accuracy: {ensemble_accuracy * 100:.2f}%')

# Print classification report for ensemble
print(classification_report(y_test, ensemble_predictions))



import matplotlib.pyplot as plt

# Access training history for BiLSTM
training_loss_bilstm = history_bilstm.history['loss']
validation_loss_bilstm = history_bilstm.history['val_loss']
training_accuracy_bilstm = history_bilstm.history['accuracy']
validation_accuracy_bilstm = history_bilstm.history['val_accuracy']

# Access training history for GRU
training_loss_gru = history_gru.history['loss']
validation_loss_gru = history_gru.history['val_loss']
training_accuracy_gru = history_gru.history['accuracy']
validation_accuracy_gru = history_gru.history['val_accuracy']

# Plot training and validation loss for BiLSTM
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(training_loss_bilstm, label='Training Loss (BiLSTM)')
plt.plot(validation_loss_bilstm, label='Validation Loss (BiLSTM)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Plot training and validation loss for GRU
plt.subplot(1, 2, 2)
plt.plot(training_loss_gru, label='Training Loss (GRU)')
plt.plot(validation_loss_gru, label='Validation Loss (GRU)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# Plot training and validation accuracy for BiLSTM
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(training_accuracy_bilstm, label='Training Accuracy (BiLSTM)')
plt.plot(validation_accuracy_bilstm, label='Validation Accuracy (BiLSTM)')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Plot training and validation accuracy for GRU
plt.subplot(1, 2, 2)
plt.plot(training_accuracy_gru, label='Training Accuracy (GRU)')
plt.plot(validation_accuracy_gru, label='Validation Accuracy (GRU)')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()
