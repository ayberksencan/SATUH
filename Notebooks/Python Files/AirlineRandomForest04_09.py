import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import tensorflow as tf
from sklearn.model_selection import train_test_split
from google.colab import drive

#Importing and starting the TPU
tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
tf.config.experimental_connect_to_cluster(tpu)
tf.tpu.experimental.initialize_tpu_system(tpu)
strategy = tf.distribute.TPUStrategy(tpu)

# Connecting to Google Drive
drive.mount('/content/drive')

# Load the preprocessed and balanced features from CSV
df_resampled_features = pd.read_csv('/content/drive/MyDrive/yuksekTez/airline_dataset/preprocessed_balanced_features.csv')

# Load the balanced labels from CSV
df_resampled_labels = pd.read_csv('/content/drive/MyDrive/yuksekTez/airline_dataset/preprocessed_balanced_labels.csv')

# Split the dataset into features (X) and labels (y)
X = df_resampled_features.values
y = df_resampled_labels['airline_sentiment'].values

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


# Create a Random Forest model
with strategy.scope():
    model = RandomForestClassifier(n_estimators=500, random_state=42)

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)


# Calculate the accuracy, precision, recall, and f1-score
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)
confusion_mat = confusion_matrix(y_test, y_pred)

# Print the results
print('Accuracy:', accuracy)
print('Classification Report:\n', classification_rep)
print('Confusion Matrix:\n', confusion_mat)

# Save predictions to a CSV file if needed
# df_pred = pd.DataFrame({"predicted_values": y_pred, "real_values": y_test})
# df_pred.to_csv('/content/drive/MyDrive/yuksekTez/airline_dataset/RandomForestPredict.csv', index=False)
