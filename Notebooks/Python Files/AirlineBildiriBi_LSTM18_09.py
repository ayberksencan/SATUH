import pandas as pd
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Bidirectional, Embedding, Dropout
import matplotlib.pyplot as plt
from google.colab import drive

# Step 3: Convert Data for TPU
resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='grpc://' + os.environ['COLAB_TPU_ADDR'])
tf.config.experimental_connect_to_cluster(resolver)
tf.tpu.experimental.initialize_tpu_system(resolver)
strategy = tf.distribute.experimental.TPUStrategy(resolver)

# Mount Google Drive
drive.mount('/content/drive')

# Step 1: Load the Preprocessed Data from Google Drive
df = pd.read_csv('/content/drive/MyDrive/yuksekTez/airline_dataset/preprocessed_data18-09.csv')
X = df.drop(columns=['label']).values
y = df['label'].values

# Step 2: Split Data into Training, Validation, and Test Sets
from sklearn.model_selection import train_test_split

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.25, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.1, random_state=42)

with strategy.scope():
    X_train_tpu = tf.convert_to_tensor(X_train, dtype=tf.float32)
    y_train_tpu = tf.convert_to_tensor(y_train, dtype=tf.int32)
    X_val_tpu = tf.convert_to_tensor(X_val, dtype=tf.float32)
    y_val_tpu = tf.convert_to_tensor(y_val, dtype=tf.int32)
    X_test_tpu = tf.convert_to_tensor(X_test, dtype=tf.float32)
    y_test_tpu = tf.convert_to_tensor(y_test, dtype=tf.int32)

# Assuming 'input_dim' is the dimension of your input features
input_dim = X_train.shape[1]

# Step 4: Define and Compile the Bi-LSTM Model
with strategy.scope():
    model = Sequential([
        Embedding(input_dim=input_dim, output_dim=64, input_length=input_dim),
        Bidirectional(LSTM(64, return_sequences=True)),
        Dropout(0.5),
        Bidirectional(LSTM(32)),
        Dense(3, activation='softmax')  # Assuming 3 classes (positive, neutral, negative)
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy', 'sparse_categorical_crossentropy'])

model.summary()

# Step 5: Train the Bi-LSTM Model
with strategy.scope():
    history = model.fit(
        X_train_tpu,
        y_train_tpu,
        epochs=15,  # Increase the number of epochs
        validation_data=(X_val_tpu, y_val_tpu),
        batch_size=64
    )

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import f1_score

# Step 6: Evaluate the Model
with strategy.scope():
    loss, accuracy = model.evaluate(X_test_tpu, y_test_tpu)
    y_pred = model.predict(X_test_tpu)
    y_pred = tf.argmax(y_pred, axis=1)
    f1 = f1_score(y_test_tpu.numpy(), y_pred.numpy(), average='weighted')
    print(f'Test Accuracy: {accuracy * 100:.2f}%')
    print(f'Weighted F1 Score: {f1:.4f}')

    # Compute and print classification report
    report = classification_report(y_test_tpu.numpy(), y_pred.numpy(), target_names=['Positive', 'Neutral', 'Negative'])
    print(f'Classification Report:\n{report}')

    # Compute and print confusion matrix
    cm = confusion_matrix(y_test_tpu.numpy(), y_pred.numpy())
    print(f'Confusion Matrix:\n{cm}')

# Step 7: Visualize the Results
# Plot training and validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plot training and validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()
