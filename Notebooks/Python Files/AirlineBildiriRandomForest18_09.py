from google.colab import drive
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# Mount Google Drive
drive.mount('/content/drive')

# Define the path to the preprocessed dataset
preprocessed_data_path = '/content/drive/MyDrive/yuksekTez/airline_dataset/preprocessed_data18-09.csv'

# Load the preprocessed data
preprocessed_df = pd.read_csv(preprocessed_data_path)

# Split data into features (X) and labels (y)
X = preprocessed_df.drop(columns=['label']).values
y = preprocessed_df['label'].values

# Split data into training, validation, and testing sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.25, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.1, random_state=42)

# Train a Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
# Train the classifier
history = rf_classifier.fit(X_train, y_train)

# Evaluate the model
accuracy = rf_classifier.score(X_test, y_test)
print(f'Accuracy: {accuracy}')

# You can also use other evaluation metrics like precision, recall, f1-score
y_pred = rf_classifier.predict(X_test)
report = classification_report(y_test, y_pred)
print(report)
