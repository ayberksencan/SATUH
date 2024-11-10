# Import the necessary libraries
import pandas as pd
from google.colab import drive

# Mount the Google Drive
drive.mount('/content/drive')

DATASET_COLUMNS = ["tweet_id", "airline_sentiment", "airline_sentiment_confidence",
       "negativereason", "negativereason_confidence", "airline",
       "airline_sentiment_gold", "name", "negativereason_gold",
       "retweet_count", "text", "tweet_coord", "tweet_created",
       "tweet_location", "user_timezone"]
DATASET_ENCODING = "ISO-8859-1"

# Set the path to the dataset file
dataset_path = '/content/drive/MyDrive/yuksekTez/airline_dataset/Tweets.csv'

# Read the dataset file into a pandas dataframe
df = pd.read_csv(dataset_path, encoding=DATASET_ENCODING, names=DATASET_COLUMNS)

# Display the first 5 rows of the dataframe
print('First 5 rows of the dataset:')
print(df.head())

# Display information about the columns
print('Columns in the dataset:')
print(df.columns)

# Display data types of the columns
print('Data types of the columns:')
print(df.dtypes)

# Display summary statistics of the numerical columns
print('Summary statistics of the numerical columns:')
print(df.describe())

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Assuming you have a DataFrame called 'df' and a column called 'column_name'
# Replace 'df' and 'column_name' with your actual DataFrame and column name

# Count the occurrences of each value in the column
value_counts = df['airline'].value_counts()

# Exclude the 'airline' value from the value counts
value_counts = value_counts.drop('airline', errors='ignore')

# Create a color map with a unique color for each value
num_unique_values = len(value_counts)
color_map = plt.cm.get_cmap('tab10', num_unique_values)

# Create a bar plot to visualize the value counts
plt.figure(figsize=(10, 6))  # Optional: Adjust the figure size
ax = value_counts.plot(kind='bar', color=color_map(np.arange(num_unique_values)))

# Set the font size for the labels
plt.rcParams['font.size'] = 14

# Exclude the 'airline' value from the plot title
ax.set_title('Value Counts of Column (Excluding "airline")')
ax.set_xlabel('Values')
ax.set_ylabel('Count')
ax.tick_params(axis='x', labelrotation=0)  # Optional: Rotate x-axis labels if needed
plt.xticks(rotation=45)
plt.show()



import pandas as pd
import matplotlib.pyplot as plt

# Assuming you have a DataFrame called 'df' and a column called 'airline'
# Replace 'df' and 'airline' with your actual DataFrame and column name

plt.figure(figsize=(8, 8), dpi=300)  # Optional: Adjust the figure size
value_counts = df['airline'].value_counts()

# Exclude the 'airline' value from the value counts
value_counts = value_counts.drop('airline', errors='ignore')

labels = value_counts.index
sizes = value_counts.values
total_count = sizes.sum()

# Set the font size for the labels
plt.rcParams['font.size'] = 14

plt.pie(sizes, labels=labels, autopct=lambda p: f'{p:.1f}%\n({p / 100 * total_count:.0f})', startangle=90)
plt.axis('equal')
plt.title('Value Counts and Percentages of Airlines')
plt.show()


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Assuming you have a DataFrame called 'df' and a column called 'airline'
# Replace 'df' and 'airline' with your actual DataFrame and column name

plt.figure(figsize=(8, 9), dpi=300)  # Optional: Adjust the figure size
value_counts = df['airline'].value_counts()

# Exclude the 'airline' value from the value counts
value_counts = value_counts.drop('airline', errors='ignore')

labels = value_counts.index
sizes = value_counts.values
total_count = sizes.sum()

# Generate a color map with a unique color for each value
num_unique_values = len(labels)
color_map = plt.cm.get_cmap('tab10', num_unique_values)
colors = color_map(np.linspace(0, 1, num_unique_values))

# Set the font size for the labels
plt.rcParams['font.size'] = 14

plt.pie(sizes, labels=labels, autopct=lambda p: f'{p:.1f}%\n({p / 100 * total_count:.0f})', startangle=90, colors=colors)
plt.axis('equal')
plt.title('Value Counts and Percentages of Airlines')
plt.show()


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6), dpi=300)  # Optional: Adjust the figure size

# Exclude the 'airline' value from the 'airline' column
df_filtered = df[df['airline'] != 'airline']

# Set the font size for the labels
plt.rcParams['font.size'] = 16

sns.countplot(data=df_filtered, x='airline', hue='airline_sentiment', palette='bright')
plt.title('Value Counts of airline_sentiment by airline')
plt.xlabel('Airline')
plt.ylabel('Count')
plt.legend(loc='upper right', title=None)
plt.xticks(rotation=45)
plt.show()




import pandas as pd
import matplotlib.pyplot as plt

# Assuming you have a DataFrame called 'df' with a column called 'airline_sentiment'
# Replace 'df' with your actual DataFrame name

plt.figure(figsize=(8, 8), dpi=300)  # Set the figsize and dpi values

value_counts = df['airline_sentiment'].value_counts()

# Exclude the 'airline_sentiment' value
value_counts = value_counts.drop('airline_sentiment')

labels = value_counts.index
sizes = value_counts.values
total_count = sizes.sum()

# Create the pie chart
patches, texts, autotexts = plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)

# Add value counts as text inside each slice
for i, (autotext, size) in enumerate(zip(autotexts, sizes)):
    autotext.set_text(f'{size} ({(size / total_count) * 100:.1f}%)')

# Set the font size for the labels
plt.rcParams['font.size'] = 16

plt.title('Value Counts of Polarity Classes')
plt.axis('equal')
plt.show()


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming you have a DataFrame called 'df' with columns 'airline' and 'airline_sentiment'
# Replace 'df' with your actual DataFrame name

plt.figure(figsize=(12, 6), dpi=300)  # Optional: Adjust the figure size

# Exclude the 'airline' value from the 'airline' column
df_filtered = df[df['airline'] != 'airline']

sns.countplot(data=df_filtered, x='airline', hue='airline_sentiment', palette='bright')
plt.title('Value Counts of airline_sentiment by Airline')
plt.xlabel('Airline')
plt.ylabel('Count')
plt.legend(loc='upper right', title=None)
plt.xticks(rotation=45)

for p in plt.gca().patches:
    height = p.get_height()
    plt.gca().annotate(f'{int(height)}', (p.get_x() + p.get_width() / 2, height), ha='center', va='bottom')

plt.show()

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming you have a DataFrame called 'df' with columns 'airline' and 'airline_sentiment'
# Replace 'df' with your actual DataFrame name

# Exclude the 'airline' value from the 'airline' column
df_filtered = df[df['airline'] != 'airline']

# Exclude the 'airline_sentiment' value from the 'airline_sentiment' column
df_filtered = df_filtered[df_filtered['airline_sentiment'] != 'airline_sentiment']

# Create a pivot table of value counts
pivot_table = df_filtered.pivot_table(index='airline_sentiment', columns='airline', aggfunc='size', fill_value=0)

plt.figure(figsize=(10, 6), dpi=300)  # Optional: Adjust the figure size

# Create the heatmap
sns.heatmap(pivot_table, cmap='YlGnBu', annot=True, fmt='d')

plt.title('Heatmap of airline sentiment')
plt.xlabel('Airline')
plt.ylabel('Sentiment')
plt.show()

