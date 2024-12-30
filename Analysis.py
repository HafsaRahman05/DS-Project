import pandas as pd
from nltk.corpus import stopwords
from collections import Counter
import matplotlib.pyplot as plt

# Load the cleaned dataset
file_path = 'cleaned_no_stopwords.csv'  # Replace with your file path
df = pd.read_csv(file_path)

# Combine text data for word frequency analysis
all_text = " ".join(df.select_dtypes(include=['object']).fillna('').values.flatten())

# Word frequency analysis
stop_words = set(stopwords.words('english'))
words = all_text.lower().split()
filtered_words = [word for word in words if word not in stop_words]
word_freq = Counter(filtered_words)

# Display the most common words
print("Top 10 Most Common Words:")
print(word_freq.most_common(10))

# Plot a bar chart of the top 10 most common words
common_words = word_freq.most_common(10)
words, counts = zip(*common_words)
plt.bar(words, counts, color='skyblue')
plt.xlabel('Words')
plt.ylabel('Frequency')
plt.title('Top 10 Most Common Words')
plt.xticks(rotation=45)
plt.show()

# Basic EDA
print("\nBasic Dataset Overview:")
print(df.describe())
print(df.info())

# Visualize the distribution of a numerical column (if available)
if not df.select_dtypes(include=['number']).empty:
    num_col = df.select_dtypes(include=['number']).columns[0]
    df[num_col].hist(bins=20, color='lightgreen', edgecolor='black')
    plt.title(f'Histogram of {num_col}')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.show()

# Save the final analyzed dataset
df.to_csv('analyzed_dataset.csv', index=False)
print("Analyzed dataset saved as 'analyzed_dataset.csv'")
