"""
Author: Ömer Ayberk ŞENCAN
Date: 31/08/2023
Description: This module implements an LSTM-based sentiment analysis model for airline tweets.
             It includes dataset preprocessing, training with TPU acceleration, and model evaluation.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
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
    labels = pd.get_dummies(df["airline_sentiment"]).values
    return padded_sequences, labels, tokenizer


def split_data(features, labels, test_size=0.25, random_state=42) -> tuple:
    """Splits the dataset into training and testing sets."""
    return train_test_split(features, labels, test_size=test_size, random_state=random_state)


def initialize_tpu():
    """Initializes TPU for distributed training."""
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    return tf.distribute.TPUStrategy(tpu)


def lr_schedule(epoch, lr):
    """Adjusts learning rate based on the epoch."""
    return lr if epoch < 20 else lr * 0.1


def build_lstm_model(strategy, vocab_size: int, embedding_dim: int, input_length: int, num_classes: int):
    """Builds and compiles an LSTM model for sentiment classification."""
    with strategy.scope():
        model = Sequential([
            Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=input_length),
            LSTM(128, return_sequences=True),
            Dropout(0.3),
            LSTM(64),
            Dense(128, activation='relu'),
            Dropout(0.5),
            BatchNormalization(),
            Dense(num_classes, activation='softmax')
        ])
        model.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
    return model


def plot_learning_curve(history):
    """Plots training and validation accuracy and loss."""
    plt.figure(figsize=(12, 5))

    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()


def evaluate_model(model, X_test, y_test):
    """Evaluates the model and prints performance metrics."""
    y_pred_probs = model.predict(X_test)
    y_pred_labels = np.argmax(y_pred_probs, axis=1)
    y_test_labels = np.argmax(y_test, axis=1)

    print("Classification Report:\n", classification_report(y_test_labels, y_pred_labels))
    print("Confusion Matrix:\n", confusion_matrix(y_test_labels, y_pred_labels))

    # Plot predicted probabilities
    plt.figure(figsize=(10, 6))
    for class_idx in range(y_test.shape[1]):
        sns.histplot(y_pred_probs[:, class_idx], label=f'Class {class_idx}', kde=True)
    plt.title('Predicted Class Probability Distribution')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Density')
    plt.legend()
    plt.show()


def main():
    """Main function to execute the LSTM sentiment analysis pipeline."""
    mount_drive()
    df = load_data()

    # Data preprocessing
    padded_sequences, labels, tokenizer = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(padded_sequences, labels)

    # TPU initialization
    strategy = initialize_tpu()

    # Build and train LSTM model
    model = build_lstm_model(strategy, vocab_size=len(tokenizer.word_index) + 1, embedding_dim=EMBEDDING_DIM,
                             input_length=MAXLEN, num_classes=labels.shape[1])
    early_stopping = EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True)
    lr_scheduler = LearningRateScheduler(lr_schedule)

    history = model.fit(X_train, y_train,
                        epochs=EPOCHS,
                        batch_size=BATCH_SIZE,
                        validation_split=0.1,
                        callbacks=[early_stopping, lr_scheduler])

    # Evaluate model
    evaluate_model(model, X_test, y_test)

    # Plot learning curves
    plot_learning_curve(history)


if __name__ == "__main__":
    main()
