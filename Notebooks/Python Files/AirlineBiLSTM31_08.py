import tensorflow as tf
tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
tf.config.experimental_connect_to_cluster(tpu)
tf.tpu.experimental.initialize_tpu_system(tpu)
strategy = tf.distribute.TPUStrategy(tpu)

# Step 1: Import Libraries and Load the Dataset
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelBinarizer

from google.colab import drive
drive.mount('/content/drive')

# Load the dataset from a CSV file
df = pd.read_csv('/content/drive/MyDrive/yuksekTez/airline_dataset/clean_text.csv')

with strategy.scope():
    # Step 2: Preprocess the Dataset
    tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
    tokenizer.fit_on_texts(df["text"])
    vocab_size = len(tokenizer.word_index) + 1
    sequences = tokenizer.texts_to_sequences(df["text"])
    maxlen = 100
    padded_sequences = pad_sequences(sequences, maxlen=maxlen, padding="post", truncating="post")
    labels = df["airline_sentiment"]

    label_binarizer = LabelBinarizer()
    labels = label_binarizer.fit_transform(labels)
    num_classes = len(label_binarizer.classes_)

# Step 3: Split the Dataset into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, labels, test_size=0.25, random_state=42)



from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler

# Learning rate schedule function
def lr_schedule(epoch, lr):
    if epoch < 20:
        return lr
    else:
        return lr * 0.1

# Early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Learning rate scheduler callback
lr_scheduler = LearningRateScheduler(lr_schedule)

with strategy.scope():
    # Define the model
    model2 = Sequential()
    model2.add(Embedding(vocab_size, 64, input_length=maxlen))  # Increased embedding size
    model2.add(Bidirectional(LSTM(128, return_sequences=True)))  # Bi-LSTM layer
    model2.add(Dropout(0.4))  # Increased dropout rate
    model2.add(Bidirectional(LSTM(64)))  # Additional Bi-LSTM layer
    model2.add(Dense(128, activation='relu'))
    model2.add(Dropout(0.5))  # Increased dropout rate
    model2.add(BatchNormalization())
    model2.add(Dense(3, activation='softmax'))

    # Compile the model
    model2.compile(optimizer=Adam(learning_rate=0.001),
                   loss='categorical_crossentropy',
                   metrics=['accuracy'])

    # Print model summary
    model2.summary()

# Train the model
history = model2.fit(X_train, y_train,
                     epochs=100,
                     batch_size=64,
                     validation_split=0.1,
                     callbacks=[early_stopping, lr_scheduler])

# Evaluate the model
y_pred = model2.predict(X_test)
y_pred_labels = np.argmax(y_pred, axis=1)
y_true_labels = np.argmax(y_test, axis=1)

print(classification_report(y_true_labels, y_pred_labels))
print(confusion_matrix(y_true_labels, y_pred_labels))
