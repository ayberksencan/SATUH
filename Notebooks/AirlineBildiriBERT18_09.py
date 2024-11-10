#!pip install pandas tensorflow transformers scikit-learn matplotlib

from google.colab import drive
import pandas as pd
from transformers import BertTokenizer, BertModel
import torch
import tensorflow as tf
import os
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertForSequenceClassification, AdamW

resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='grpc://' + os.environ['COLAB_TPU_ADDR'])
tf.config.experimental_connect_to_cluster(resolver)
tf.tpu.experimental.initialize_tpu_system(resolver)
strategy = tf.distribute.TPUStrategy(resolver)

# Load preprocessed data
# Mount Google Drive
drive.mount('/content/drive')

# Step 1: Load the Preprocessed Data from Google Drive
# Load preprocessed data
preprocessed_df = pd.read_csv('/content/drive/MyDrive/yuksekTez/airline_dataset/preprocessed_data18-09.csv')


