import tensorflow as tf
# Connect to TPU
tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
tf.config.experimental_connect_to_cluster(tpu)
tf.tpu.experimental.initialize_tpu_system(tpu)
strategy = tf.distribute.TPUStrategy(tpu)

import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout, MaxPooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from google.colab import drive

# Connect to Google Drive
drive.mount('/content/drive')

# Load the preprocessed and balanced dataset
df_features = pd.read_csv('/content/drive/MyDrive/yuksekTez/airline_dataset/preprocessed_balanced_features.csv')
df_labels = pd.read_csv('/content/drive/MyDrive/yuksekTez/airline_dataset/preprocessed_balanced_labels.csv')

X = df_features.values
y = df_labels['airline_sentiment'].values

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
input_dim = X_train.shape[1]

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, Flatten, Dense, Dropout
# Create a simple CNN model
with strategy.scope():
    model = keras.Sequential([
        tf.keras.layers.Embedding(input_dim=input_dim, output_dim=64, input_length=X_train.shape[1]),
        tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding="same", activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.GlobalMaxPooling1D(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(units=64, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(units=3, activation="softmax")
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Print the model summary
model.summary()

from sklearn.preprocessing import LabelEncoder

# Initialize the label encoder
label_encoder = LabelEncoder()

# Encode your string labels into integers
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Convert your data to TensorFlow tensors
X_train_tensor = tf.convert_to_tensor(X_train, dtype=tf.float32)
y_train_tensor = tf.convert_to_tensor(y_train_encoded, dtype=tf.int32)
X_test_tensor = tf.convert_to_tensor(X_test, dtype=tf.float32)
y_test_tensor = tf.convert_to_tensor(y_test_encoded, dtype=tf.int32)

from tensorflow.keras.callbacks import EarlyStopping

# Define early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model using TensorFlow tensors and include early stopping
with strategy.scope():
    history = model.fit(X_train_tensor, y_train_tensor,
              epochs=100,
              batch_size=32,
              validation_data=(X_test_tensor, y_test_tensor),
              callbacks=[early_stopping])

# Evaluate the model
loss, accuracy = model.evaluate(X_test_tensor, y_test_tensor)
print('Test accuracy:', accuracy)

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

# After training the model, you can make predictions
y_pred_probs = model.predict(X_test_tensor)

# Convert predicted probabilities to class labels
y_pred_labels = np.argmax(y_pred_probs, axis=1)

# Define the class names corresponding to your sentiment classes
class_names = ["negative", "neutral", "positive"]

# Generate the classification report
classification_report_str = classification_report(y_test_tensor, y_pred_labels, target_names=class_names)
print('Classification Report:\n', classification_report_str)

# Generate the confusion matrix
confusion_mat = confusion_matrix(y_test, y_pred_labels)

# Print or plot the confusion matrix as needed
print('Confusion Matrix:\n', confusion_mat)

#!pip install matplotlib
import matplotlib.pyplot as plt

def plot_learning_curve(history):
    # Plot training & validation accuracy values
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

# Assuming 'history' is the variable storing your model's training history
plot_learning_curve(history)
