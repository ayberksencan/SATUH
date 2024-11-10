import tensorflow as tf
tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
tf.config.experimental_connect_to_cluster(tpu)
tf.tpu.experimental.initialize_tpu_system(tpu)
strategy = tf.distribute.TPUStrategy(tpu)
#!pip install imbalanced-learn

# Step 1: Import Libraries and Load the Dataset
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Install imbalanced-learn library
#!pip install imbalanced-learn

from google.colab import drive
drive.mount('/content/drive')

# Load the dataset from a CSV file
df = pd.read_csv('/content/drive/MyDrive/yuksekTez/airline_dataset/cleaned_text.csv')

# Apply TPU scope for preprocessing
with strategy.scope():
    # Tokenize and preprocess the dataset
    tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
    tokenizer.fit_on_texts(df["text"])
    vocab_size = len(tokenizer.word_index) + 1
    sequences = tokenizer.texts_to_sequences(df["text"])
    maxlen = 100
    padded_sequences = pad_sequences(sequences, maxlen=maxlen, padding="post", truncating="post")
    labels = pd.get_dummies(df["airline_sentiment"]).values


# Split the Dataset into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, labels, test_size=0.20, random_state=42)

# Apply SMOTE only to the training set
smote = SMOTE(sampling_strategy='auto', random_state=42, k_neighbors=5)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Define the Bi-LSTM model within the TPU scope
with strategy.scope():
    model1 = tf.keras.Sequential([
        tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=32, input_length=maxlen),  # Reduced embedding dimensions
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(16, return_sequences=True)),   # Reduced LSTM units
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(16)),
        tf.keras.layers.Dense(units=16, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.01)),  # Increased L2 regularization
        tf.keras.layers.Dropout(0.6),  # Increased dropout rate
        tf.keras.layers.Dense(units=3, activation="softmax")
    ])

    model1.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),  # Reduced learning rate further
                   loss="categorical_crossentropy",
                   metrics=["accuracy"])

# Define early stopping callback with increased patience
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)  # Increased patience

# Step 5: Compile the Model and Define the Optimizer and Loss Function
model1.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

model1.summary()

# Train the Model with resampled data
history = model1.fit(X_resampled, y_resampled, epochs=150, batch_size=64, validation_split=0.2, callbacks=[early_stopping])  # Increased epochs

# Evaluate the Model on the Testing Set
y_pred = model1.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)
y_test = np.argmax(y_test, axis=1)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# Define plotting functions
def plot_learning_curve(history):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

# Plot learning curves
plot_learning_curve(history)
