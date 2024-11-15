"""
Author: Ömer Ayberk ŞENCAN
Date: 03/05/2023
Description: This module utilizes a CNN model for sentiment analysis on airline tweets.
             It includes dataset preprocessing, oversampling with SMOTE, and training
             on a TPU for optimized performance.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from google.colab import drive
import matplotlib.pyplot as plt

# Constants
DATASET_PATH = '/content/drive/MyDrive/yuksekTez/airline_dataset/cleaned_text.csv'
VOCAB_SIZE = 10000
MAXLEN = 100
EMBEDDING_DIM = 64
BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 0.001
PATIENCE = 10

def mount_drive():
    """Mounts Google Drive for data access."""
    drive.mount('/content/drive')


def load_data(path: str = DATASET_PATH) -> pd.DataFrame:
    """Loads the dataset from a CSV file."""
    return pd.read_csv(path)


def preprocess_data(df: pd.DataFrame) -> tuple:
    """Tokenizes text data and encodes labels for training."""
    tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
    tokenizer.fit_on_texts(df["text"])
    sequences = tokenizer.texts_to_sequences(df["text"])
    padded_sequences = pad_sequences(sequences, maxlen=MAXLEN, padding="post", truncating="post")
    labels = pd.get_dummies(df["airline_sentiment"]).values
    return padded_sequences, labels, tokenizer


def initialize_tpu():
    """Initializes TPU for distributed training."""
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    return tf.distribute.TPUStrategy(tpu)


def build_model(strategy, vocab_size: int, embedding_dim: int, input_length: int, num_classes: int):
    """Builds and compiles the CNN model with dropout and batch normalization."""
    with strategy.scope():
        model = Sequential([
            Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=input_length),
            Conv1D(filters=64, kernel_size=3, padding="same", activation="relu"),
            BatchNormalization(),
            GlobalMaxPooling1D(),
            Dropout(0.5),
            Dense(64, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            Dropout(0.5),
            Dense(num_classes, activation="softmax")
        ])
        model.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
                      loss="categorical_crossentropy",
                      metrics=["accuracy"])
    return model


def resample_data(X_train, y_train):
    """Applies SMOTE to the training data."""
    smote = SMOTE(sampling_strategy='auto', random_state=42, k_neighbors=5)
    return smote.fit_resample(X_train, y_train)


def plot_learning_curve(history):
    """Plots learning curves for accuracy and loss."""
    plt.figure(figsize=(12, 5))
    # Accuracy Plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Loss Plot
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
    """Main function to execute the CNN sentiment analysis pipeline."""
    mount_drive()
    df = load_data()

    # Data preprocessing
    padded_sequences, labels, tokenizer = preprocess_data(df)
    X_train, X_test, y_train, y_test = train_test_split(padded_sequences, labels, test_size=0.25, random_state=42)

    # Data resampling using SMOTE
    X_resampled, y_resampled = resample_data(X_train, y_train)

    # TPU initialization
    strategy = initialize_tpu()

    # Model building and training
    model = build_model(strategy, vocab_size=len(tokenizer.word_index) + 1, embedding_dim=EMBEDDING_DIM,
                        input_length=MAXLEN, num_classes=labels.shape[1])

    early_stopping = EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True)

    history = model.fit(X_resampled, y_resampled, epochs=EPOCHS, batch_size=BATCH_SIZE,
                        validation_split=0.1, callbacks=[early_stopping])

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
