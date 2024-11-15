"""
Author: Ömer Ayberk ŞENCAN
Date: 06/09/2024
Description: This module implements a CNN-based sentiment analysis model for airline tweets.
             It uses preprocessed and balanced data, leveraging TensorFlow and TPU for training.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from google.colab import drive

# Constants
FEATURES_PATH = '/content/drive/MyDrive/yuksekTez/airline_dataset/preprocessed_balanced_features.csv'
LABELS_PATH = '/content/drive/MyDrive/yuksekTez/airline_dataset/preprocessed_balanced_labels.csv'
EMBEDDING_DIM = 64
BATCH_SIZE = 32
EPOCHS = 100
PATIENCE = 10


def mount_drive():
    """Mounts Google Drive for data access."""
    drive.mount('/content/drive')


def load_data(features_path: str = FEATURES_PATH, labels_path: str = LABELS_PATH) -> tuple:
    """Loads preprocessed feature and label datasets."""
    features = pd.read_csv(features_path).values
    labels = pd.read_csv(labels_path)['airline_sentiment'].values
    return features, labels


def split_data(features, labels, test_size: float = 0.25, random_state: int = 42) -> tuple:
    """Splits the dataset into training and testing sets."""
    return train_test_split(features, labels, test_size=test_size, random_state=random_state)


def initialize_tpu():
    """Initializes TPU for distributed training."""
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    return tf.distribute.TPUStrategy(tpu)


def encode_labels(labels) -> tuple:
    """Encodes string labels into integer format."""
    encoder = LabelEncoder()
    encoded_labels = encoder.fit_transform(labels)
    return encoded_labels, encoder


def build_model(strategy, input_dim: int, embedding_dim: int, num_classes: int):
    """Builds and compiles a CNN model for sentiment classification."""
    with strategy.scope():
        model = Sequential([
            Embedding(input_dim=input_dim, output_dim=embedding_dim, input_length=input_dim),
            Conv1D(filters=64, kernel_size=3, padding="same", activation="relu"),
            BatchNormalization(),
            GlobalMaxPooling1D(),
            Dropout(0.5),
            Dense(64, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            Dropout(0.5),
            Dense(num_classes, activation="softmax")
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def plot_learning_curve(history):
    """Plots learning curves for accuracy and loss."""
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


def evaluate_model(model, X_test, y_test, encoder):
    """Evaluates the trained model and displays metrics."""
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f'Test accuracy: {accuracy:.4f}')

    y_pred_probs = model.predict(X_test)
    y_pred_labels = np.argmax(y_pred_probs, axis=1)

    print('Classification Report:\n', classification_report(y_test, y_pred_labels, target_names=encoder.classes_))
    print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred_labels))


def main():
    """Main function to execute the CNN sentiment analysis pipeline."""
    mount_drive()
    features, labels = load_data()
    X_train, X_test, y_train, y_test = split_data(features, labels)

    # TPU initialization
    strategy = initialize_tpu()

    # Encode labels
    y_train_encoded, encoder = encode_labels(y_train)
    y_test_encoded = encoder.transform(y_test)

    # Convert to tensors
    X_train_tensor = tf.convert_to_tensor(X_train, dtype=tf.float32)
    y_train_tensor = tf.convert_to_tensor(y_train_encoded, dtype=tf.int32)
    X_test_tensor = tf.convert_to_tensor(X_test, dtype=tf.float32)
    y_test_tensor = tf.convert_to_tensor(y_test_encoded, dtype=tf.int32)

    # Build and train model
    model = build_model(strategy, input_dim=X_train.shape[1], embedding_dim=EMBEDDING_DIM, num_classes=len(encoder.classes_))
    early_stopping = EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True)

    history = model.fit(X_train_tensor, y_train_tensor,
                        epochs=EPOCHS,
                        batch_size=BATCH_SIZE,
                        validation_data=(X_test_tensor, y_test_tensor),
                        callbacks=[early_stopping])

    # Evaluate model
    evaluate_model(model, X_test_tensor, y_test_tensor, encoder)

    # Plot learning curve
    plot_learning_curve(history)


if __name__ == "__main__":
    main()
