# bert_sentiment_analysis.py

"""
Author: Ömer Ayberk ŞENCAN
Date: 31/08/2023
Description: This module leverages BERT for sentiment classification of airline tweets.
             It includes functions for loading, preprocessing, tokenizing, and training a
             BERT-based model on the dataset. The code is optimized to use TPU for improved performance.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from google.colab import drive

# Constants
DATASET_PATH = '/content/drive/MyDrive/yuksekTez/airline_dataset/clean_text.csv'
LABEL_MAPPING = {'negative': 0, 'neutral': 1, 'positive': 2}
BERT_MODEL_NAME = 'bert-base-uncased'
MAX_LENGTH = 128
BATCH_SIZE = 32
EPOCHS = 3
LEARNING_RATE = 2e-5


def mount_drive():
    """Mounts Google Drive for data access."""
    drive.mount('/content/drive')


def load_data(path: str = DATASET_PATH) -> pd.DataFrame:
    """Loads the dataset from a CSV file."""
    return pd.read_csv(path)


def preprocess_data(df: pd.DataFrame) -> tuple:
    """Preprocesses the dataset by encoding labels and splitting into train and test sets."""
    y = [LABEL_MAPPING[label] for label in df['airline_sentiment']]
    X_train, X_test, y_train, y_test = train_test_split(df['clean_text'], y, test_size=0.25, random_state=42)
    return X_train, X_test, y_train, y_test


def tokenize_data(tokenizer, texts, max_length=MAX_LENGTH) -> dict:
    """Tokenizes and encodes text data for BERT input."""
    return tokenizer.batch_encode_plus(texts.tolist(), max_length=max_length, padding=True, truncation=True, return_tensors='tf')


def initialize_tpu():
    """Initializes TPU for distributed training."""
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    return tf.distribute.TPUStrategy(tpu)


def create_model(strategy) -> TFBertForSequenceClassification:
    """Creates and compiles a BERT model for sequence classification."""
    with strategy.scope():
        model = TFBertForSequenceClassification.from_pretrained(BERT_MODEL_NAME, num_labels=len(LABEL_MAPPING))
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )
    return model


def train_model(model, train_data, strategy, epochs=EPOCHS):
    """Trains the BERT model using TPU strategy."""
    with strategy.scope():
        for epoch in range(epochs):
            epoch_loss_avg = tf.keras.metrics.Mean()
            for batch_inputs, batch_labels in train_data:
                loss_value = train_step(model, batch_inputs, batch_labels)
                epoch_loss_avg.update_state(loss_value)
            print(f"Epoch {epoch + 1}: Loss {epoch_loss_avg.result():.4f}")


@tf.function
def train_step(model, batch_inputs, batch_labels):
    """Performs a single training step."""
    with tf.GradientTape() as tape:
        logits = model(batch_inputs)[0]
        loss_value = tf.keras.losses.sparse_categorical_crossentropy(batch_labels, logits, from_logits=True)
    grads = tape.gradient(loss_value, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss_value


def evaluate_model(model, test_data, y_test):
    """Evaluates the model on the test set."""
    eval_results = model.evaluate([test_data['input_ids'], test_data['attention_mask']], y_test)
    print("Evaluation results:", eval_results)


if __name__ == "__main__":
    # Step-by-step execution
    mount_drive()
    df = load_data()
    X_train, X_test, y_train, y_test = preprocess_data(df)

    strategy = initialize_tpu()
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)

    X_train_encoded = tokenize_data(tokenizer, X_train)
    X_test_encoded = tokenize_data(tokenizer, X_test)

    model = create_model(strategy)
    train_model(model, X_train_encoded, strategy)
    evaluate_model(model, X_test_encoded, y_test)
