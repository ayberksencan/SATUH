#!pip install --upgrade pip setuptools
#!pip install cython
#!pip install tokenizers --no-build-isolation
#!pip cache purge
#!pip install transformers

import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification
tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
tf.config.experimental_connect_to_cluster(tpu)
tf.tpu.experimental.initialize_tpu_system(tpu)
strategy = tf.distribute.TPUStrategy(tpu)

# Step 1: Import Libraries and Load the Dataset
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from google.colab import drive
drive.mount('/content/drive')

# Load the dataset from a CSV file
df = pd.read_csv('/content/drive/MyDrive/yuksekTez/airline_dataset/clean_text.csv')


with strategy.scope():
    # Step 2: Preprocess the Dataset

    # Create a label mapping dictionary
    label_mapping = {'negative': 0, 'neutral': 1, 'positive': 2}

    # Convert sentiment labels to integers using the label mapping
    y = [label_mapping[label] for label in df['airline_sentiment']]

    # Filter out rows with missing or NaN values in the 'clean_text' column
    #df = df.dropna(subset=['clean_text'])

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(df['clean_text'], y, test_size=0.25, random_state=42)

    # Load the BERT Tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

   # Tokenize and encode the sentences
    X_train_encoded = tokenizer.batch_encode_plus(X_train.tolist(), max_length=128, padding=True, truncation=True, return_tensors='tf')
    X_test_encoded = tokenizer.batch_encode_plus(X_test.tolist(), max_length=128, padding=True, truncation=True, return_tensors='tf')

# Load pre-trained BERT model
model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5), loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

@tf.function
def train_step(batch_inputs, batch_labels):
    with tf.GradientTape() as tape:
        logits = model(batch_inputs)[0]  # Get the logits from the model output
        loss_value = tf.keras.losses.sparse_categorical_crossentropy(batch_labels, logits, from_logits=True)
    grads = tape.gradient(loss_value, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss_value



with strategy.scope():
    # Train the model
    for epoch in range(3):
        epoch_loss_avg = tf.keras.metrics.Mean()
        for batch_inputs, batch_labels in train_dataset:
            loss_value = train_step(batch_inputs, batch_labels)
            epoch_loss_avg.update_state(loss_value)

        print("Epoch {}: Loss {:.4f}".format(epoch, epoch_loss_avg.result()))


# Evaluate the model
eval_results = model.evaluate([X_test_encoded['input_ids'], X_test_encoded['attention_mask']], y_test)
print("Evaluation results:", eval_results)
