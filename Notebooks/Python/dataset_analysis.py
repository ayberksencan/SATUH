"""
Author: Ömer Ayberk ŞENCAN
Date: 28/04/2023
Description: This module is designed for inspecting and analyzing the airline dataset.
             It provides functions to load, analyze, and visualize various aspects
             of the dataset, including basic statistics and visual representations.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from google.colab import drive

# Constants
DATASET_COLUMNS = [
    "tweet_id", "airline_sentiment", "airline_sentiment_confidence",
    "negativereason", "negativereason_confidence", "airline",
    "airline_sentiment_gold", "name", "negativereason_gold",
    "retweet_count", "text", "tweet_coord", "tweet_created",
    "tweet_location", "user_timezone"
]
DATASET_ENCODING = "ISO-8859-1"
DATASET_PATH = '/content/drive/MyDrive/yuksekTez/airline_dataset/Tweets.csv'


def load_data(path: str = DATASET_PATH, encoding: str = DATASET_ENCODING) -> pd.DataFrame:
    """Loads dataset into a DataFrame."""
    drive.mount('/content/drive')
    df = pd.read_csv(path, encoding=encoding, names=DATASET_COLUMNS)
    return df


def display_basic_info(df: pd.DataFrame) -> None:
    """Displays basic information about the DataFrame."""
    print('First 5 rows of the dataset:')
    print(df.head())
    print('\nColumns in the dataset:')
    print(df.columns)
    print('\nData types of the columns:')
    print(df.dtypes)
    print('\nSummary statistics of the numerical columns:')
    print(df.describe())


def plot_airline_value_counts(df: pd.DataFrame) -> None:
    """Plots bar chart of airline value counts."""
    value_counts = df['airline'].value_counts().drop('airline', errors='ignore')
    plt.figure(figsize=(10, 6))
    color_map = plt.cm.get_cmap('tab10', len(value_counts))
    value_counts.plot(kind='bar', color=color_map(np.arange(len(value_counts))))
    plt.title('Airline Value Counts')
    plt.xlabel('Airline')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.show()


def plot_airline_pie_chart(df: pd.DataFrame) -> None:
    """Plots a pie chart of airline distributions."""
    value_counts = df['airline'].value_counts().drop('airline', errors='ignore')
    labels, sizes = value_counts.index, value_counts.values
    total_count = sizes.sum()
    plt.figure(figsize=(8, 8), dpi=300)
    plt.pie(sizes, labels=labels, autopct=lambda p: f'{p:.1f}%\n({int(p / 100 * total_count)})', startangle=90)
    plt.axis('equal')
    plt.title('Airline Distribution')
    plt.show()


def plot_airline_sentiment_count(df: pd.DataFrame) -> None:
    """Plots count of airline sentiments by airline."""
    df_filtered = df[df['airline'] != 'airline']
    plt.figure(figsize=(12, 6), dpi=300)
    sns.countplot(data=df_filtered, x='airline', hue='airline_sentiment', palette='bright')
    plt.title('Airline Sentiment Counts by Airline')
    plt.xlabel('Airline')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.legend(title='Sentiment')
    plt.show()


def plot_sentiment_heatmap(df: pd.DataFrame) -> None:
    """Plots heatmap of airline sentiment by airline."""
    df_filtered = df[(df['airline'] != 'airline') & (df['airline_sentiment'] != 'airline_sentiment')]
    pivot_table = df_filtered.pivot_table(index='airline_sentiment', columns='airline', aggfunc='size', fill_value=0)
    plt.figure(figsize=(10, 6), dpi=300)
    sns.heatmap(pivot_table, cmap='YlGnBu', annot=True, fmt='d')
    plt.title('Heatmap of Airline Sentiments')
    plt.xlabel('Airline')
    plt.ylabel('Sentiment')
    plt.show()


# Main function to run analysis (for testing and execution in scripts)
if __name__ == "__main__":
    df = load_data()
    display_basic_info(df)
    plot_airline_value_counts(df)
    plot_airline_pie_chart(df)
    plot_airline_sentiment_count(df)
    plot_sentiment_heatmap(df)
