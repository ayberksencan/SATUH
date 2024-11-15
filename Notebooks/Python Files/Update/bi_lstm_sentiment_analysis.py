"""
Author: Ömer Ayberk ŞENCAN
Date: 29/08/2023
Description: This module utilizes a Bidirectional LSTM model to classify sentiments of airline tweets.
             It includes steps for loading, preprocessing, resampling, training, and evaluating the model,
             optimized for TPU usage.

"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
from google.colab import drive

# Constants
DATASET_PATH = '/content/drive/MyDrive/yuksekTez/airline_dataset/cleaned_text.csv'
VOCAB_SIZE = 10000
MAXLEN = 100
EMBEDDING_DIM = 32
BATCH_SIZE = 64
EPOCHS = 150
LEARNING_RATE = 0.0001
PATIENCE = 30

def mount_drive():
    """Mounts Google Drive for data access."""
    drive.mount('/content/drive')


def load_data(path: str = DATASET_PATH) -> pd.DataFrame:
    """Loads the dataset from a CSV file."""
    return pd.read_csv(path)


def preprocess_data(df: pd.DataFrame) -> tuple:
    """Tokenizes text data and encodes labels."""
    tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
    tokenizer.fit_on_texts(df["text"])
    sequences = tokenizer.texts_to_sequences(df["text"])
    padded_sequences = pad_sequences(sequences, maxlen=MAXLEN, padding="post", truncating="post")
    labels = pd.get_dummies(df["airline_sentiment"]).values
    return padded_sequences, labels, tokenizer


def resample_data(X_train, y_train) -> tuple:
    """Applies SMOTE to the training data."""
    smote = SMOTE(sampling_strategy='auto', random_state=42, k_neighbors=5)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    return X_resampled, y_resampled


def initialize_tpu():
    """Initializes TPU for distributed training."""
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    return tf.distribute.TPUStrategy(tpu)


def build_model(strategy, vocab_size: int, embedding_dim: int, input_length: int):
    """Builds and compiles the Bi-LSTM model."""
    with strategy.scope():
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=input_length),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(16, return_sequences=True)),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(16)),
            tf.keras.layers.Dense(units=16, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            tf.keras.layers.Dropout(0.6),
            tf.keras.layers.Dense(units=3, activation="softmax")
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                      loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def plot_learning_curve(history):
    """Plots learning curves for accuracy and loss."""
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()


def main():
    """Main function to execute the Bi-LSTM sentiment analysis pipeline."""
    mount_drive()
    df = load_data()

    # Data preprocessing
    padded_sequences, labels, tokenizer = preprocess_data(df)
    X_train, X_test, y_train, y_test = train_test_split(padded_sequences, labels, test_size=0.2, random_state=42)
    X_resampled, y_resampled = resample_data(X_train, y_train)

    # TPU initialization
    strategy = initialize_tpu()

    # Model building and training
    model = build_model(strategy, vocab_size=VOCAB_SIZE + 1, embedding_dim=EMBEDDING_DIM, input_length=MAXLEN)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True)

    history = model.fit(X_resampled, y_resampled, epochs=EPOCHS, batch_size=BATCH_SIZE,
                        validation_split=0.2, callbacks=[early_stopping])

    # Model evaluation
    y_pred = model.predict(X_test)
    y_pred = np.argmax(y_pred, axis=1)
    y_test = np.argmax(y_test, axis=1)
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

    # Plot learning curves
    plot_learning_curve(history)


if __name__ == "__main__":
    main()
