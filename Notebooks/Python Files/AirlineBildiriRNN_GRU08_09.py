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

from tensorflow.keras.callbacks import EarlyStopping
# Define the model-building function (to be used inside the strategy.scope())
def build_model():
    model = Sequential([
        GRU(64, input_shape=(X.shape[1], X.shape[2])),
        Dense(3, activation='softmax')  # 3 output neurons for 3 classes, softmax for multi-class classification
    ])
    return model

# Use the strategy scope to distribute the training
with strategy.scope():
    model = build_model()
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

with strategy.scope():
   # Define EarlyStopping callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

    # Train the model with early stopping
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=64, callbacks=[early_stopping])

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

