import tensorflow as tf
# Connect to TPU
tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
tf.config.experimental_connect_to_cluster(tpu)
tf.tpu.experimental.initialize_tpu_system(tpu)
strategy = tf.distribute.TPUStrategy(tpu)

import pandas as pd
import numpy as np
from google.colab import drive
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
import tensorflow as tf

# Connecting to Google Drive
drive.mount('/content/drive')

# Load the preprocessed and balanced features from CSV
df_resampled_features = pd.read_csv('/content/drive/MyDrive/yuksekTez/airline_dataset/preprocessed_balanced_features.csv')
# Load the balanced labels from CSV
df_resampled_labels = pd.read_csv('/content/drive/MyDrive/yuksekTez/airline_dataset/preprocessed_balanced_labels.csv')

# Convert the dataframes to numpy arrays
X = df_resampled_features.values
y = df_resampled_labels['airline_sentiment'].values

from sklearn.model_selection import train_test_split

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Naive Bayes classifier with cross-validation
with strategy.scope():
    model = MultinomialNB()
    scores = cross_val_score(model, X_train, y_train, cv=5)
    print("Cross-Validation Scores:", scores)
    print("Mean CV Score:", np.mean(scores))

# Train the model on the entire training set
with strategy.scope():
    model.fit(X_train, y_train)

# Test the classifier on the test set
with strategy.scope():
    y_pred = model.predict(X_test)

# Evaluate the performance of the classifier
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
