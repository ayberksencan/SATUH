"""
Author: Ömer Ayberk ŞENCAN
Date: 31/08/2023
Description: This module implements a Bi-LSTM model for sentiment analysis on airline tweets.
             It includes dataset preprocessing, a custom learning rate scheduler, and training
             on a TPU for optimized performance.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelBinarizer
from google.colab import drive

# Constants
DATASET_PATH = '/content/drive/MyDrive/yuksekTez/airline_dataset/clean_text.csv'
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

    label_binarizer = LabelBinarizer()
    labels = label_binarizer.fit_transform(df["airline_sentiment"])
    return padded_sequences, labels, tokenizer


def initialize_tpu():
    """Initializes TPU for distributed training."""
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    return tf.distribute.TPUStrategy(tpu)


def lr_schedule(epoch, lr):
    """Learning rate schedule function that reduces the learning rate after 20 epochs."""
    return lr if epoch < 20 else lr * 0.1


def build_model(strategy, vocab_size: int, embedding_dim: int, input_length: int, num_classes: int):
    """Builds and compiles the Bi-LSTM model with dropout and batch normalization."""
    with strategy.scope():
        model = Sequential([
            Embedding(vocab_size, embedding_dim, input_length=input_length),
            Bidirectional(LSTM(128, return_sequences=True)),
            Dropout(0.4),
            Bidirectional(LSTM(64)),
            Dense(128, activation='relu'),
            Dropout(0.5),
            BatchNormalization(),
            Dense(num_classes, activation='softmax')
        ])
        model.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
    return model


def main():
    """Main function to execute the Bi-LSTM sentiment analysis pipeline."""
    mount_drive()
    df = load_data()

    # Data preprocessing
    padded_sequences, labels, tokenizer = preprocess_data(df)
    X_train, X_test, y_train, y_test = train_test_split(padded_sequences, labels, test_size=0.25, random_state=42)

    # TPU initialization
    strategy = initialize_tpu()

    # Model building and training
    model = build_model(strategy, vocab_size=len(tokenizer.word_index) + 1, embedding_dim=EMBEDDING_DIM,
                        input_length=MAXLEN, num_classes=labels.shape[1])

    early_stopping = EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True)
    lr_scheduler = LearningRateScheduler(lr_schedule)

    history = model.fit(X_train, y_train,
                        epochs=EPOCHS,
                        batch_size=BATCH_SIZE,
                        validation_split=0.1,
                        callbacks=[early_stopping, lr_scheduler])

    # Model evaluation
    y_pred = model.predict(X_test)
    y_pred_labels = np.argmax(y_pred, axis=1)
    y_true_labels = np.argmax(y_test, axis=1)

    print(classification_report(y_true_labels, y_pred_labels))
    print(confusion_matrix(y_true_labels, y_pred_labels))


if __name__ == "__main__":
    main()
