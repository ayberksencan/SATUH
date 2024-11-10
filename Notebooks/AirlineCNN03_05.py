import tensorflow as tf
tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
tf.config.experimental_connect_to_cluster(tpu)
tf.tpu.experimental.initialize_tpu_system(tpu)
strategy = tf.distribute.TPUStrategy(tpu)
#!pip install imbalanced-learn

# Step 1: Import Libraries and Load the Dataset
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from google.colab import drive
drive.mount('/content/drive')

# Load the dataset from a CSV file
df = pd.read_csv('/content/drive/MyDrive/yuksekTez/airline_dataset/cleaned_text.csv')

with strategy.scope():
  # Step 2: Preprocess the Dataset
  tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
  tokenizer.fit_on_texts(df["text"])
  vocab_size = len(tokenizer.word_index) + 1
  sequences = tokenizer.texts_to_sequences(df["text"])
  maxlen = 100
  padded_sequences = pad_sequences(sequences, maxlen=maxlen, padding="post", truncating="post")
  labels = pd.get_dummies(df["airline_sentiment"]).values
  print(vocab_size)

# Step 3: Split the Dataset into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, labels, test_size=0.25, random_state=42)

with strategy.scope():
    model1 = tf.keras.Sequential([
        tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=64, input_length=maxlen),
        tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding="same", activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.GlobalMaxPooling1D(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(units=64, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(units=3, activation="softmax")
    ])

    model1.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                   loss="categorical_crossentropy",
                   metrics=["accuracy"])

# Define early stopping callback with reduced patience
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)


model1.summary()

# Step 5: Compile the Model and Define the Optimizer and Loss Function
model1.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

model1.summary()

# Step 5,5: SMOTE
from imblearn.over_sampling import SMOTE

# Initialize SMOTE with hyperparameters
smote = SMOTE(
    sampling_strategy='auto',  # Adjust based on the imbalance ratio
    random_state=42,           # Set a random seed for reproducibility
    k_neighbors=5,             # Number of nearest neighbors to consider
)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
X_resampled_test, y_resampled_test = smote.fit_resample(X_test, y_test)

with strategy.scope():
  # Step 6: Train the Model
  history = model1.fit(X_train, y_train, epochs=100, batch_size=64, validation_split=0.1, callbacks=[early_stopping])

# Step 7: Evaluate the Model on the Testing Set
y_pred = model1.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)
y_test = np.argmax(y_test, axis=1)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

import matplotlib.pyplot as plt

def plot_learning_curve(history):
    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.show()

# Assuming 'history' is the variable storing your model's training history
plot_learning_curve(history)


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


"""def create_detailed_cnn_model(input_shape, num_classes):
    model2 = Sequential()

    # Convolutional Layer 1
    model2.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model2.add(MaxPooling2D(pool_size=(2, 2)))

    # Convolutional Layer 2
    model2.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model2.add(MaxPooling2D(pool_size=(2, 2)))

    # Convolutional Layer 3
    model2.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model2.add(MaxPooling2D(pool_size=(2, 2)))

    # Flatten
    model2.add(Flatten())

    # Fully Connected Layer 1
    model2.add(Dense(512, activation='relu'))
    model2.add(Dropout(0.5))

    # Fully Connected Layer 2
    model2.add(Dense(256, activation='relu'))
    model2.add(Dropout(0.5))

    # Output Layer
    model2.add(Dense(num_classes, activation='softmax'))

    return model2

# Define input shape and number of classes
input_shape = (None, 10000, 3)  # Adjust the input shape based on your data
num_classes = 3  # Change this to the number of classes in your problem

# Create the model
model2 = create_detailed_cnn_model(input_shape, num_classes)

# Compile the model
model2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Print model summary
model2.summary()"""

with strategy.scope():
  # Step 6: Train the Model
  history = model1.fit(X_resampled, y_resampled, epochs=100, batch_size=64, validation_split=0.1, callbacks=[early_stopping])

# Step 7: Evaluate the Model on the Testing Set
y_pred = model1.predict(X_resampled_test)
y_pred = np.argmax(y_pred, axis=1)
y_test = np.argmax(y_resampled_test, axis=1)
print(classification_report(y_resampled_test, y_pred))
print(confusion_matrix(y_resampled_test, y_pred))
