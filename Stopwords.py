import pandas as pd
import nltk
from nltk.corpus import stopwords


# Load the cleaned data
df_cleaned = pd.read_csv('new_file.csv')

# Download stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))






# Combine all text data into one string
all_text = " ".join(
    df_cleaned[col].astype(str).sum() for col in df_cleaned.columns if df_cleaned[col].dtype == 'object'
)

# Tokenize the text (split into words)
words = all_text.split()

# Count the total number of words
total_word_count = len(words)

print("Total Word Count in the File:", total_word_count)






# Define a function to remove stopwords
def remove_stopwords(text):
    if isinstance(text, str):
        words = text.split()
        filtered_words = [word for word in words if word.lower() not in stop_words]
        return " ".join(filtered_words)
    return text


# Step 4: Apply stopwords removal to all columns
df_cleaned = df_cleaned.applymap(
    lambda text: remove_stopwords(text) if isinstance(text, str) else text
)





# Combine all text data into one string
all_text = " ".join(
    df_cleaned[col].astype(str).sum() for col in df_cleaned.columns if df_cleaned[col].dtype == 'object'
)

# Tokenize the text (split into words)
words = all_text.split()

# Count the total number of words
total_word_count = len(words)

print("Total Word Count in the File:", total_word_count)








import re
characters_to_remove = ".,;!@#%&*()-_=+<>?/\\[]{}\" """
def remove_special_characters(text):
    if isinstance(text, str):  # Ensure the input is a string
        translation_table = str.maketrans('','', characters_to_remove)
        cleaned_text=text.translate(translation_table)
        return cleaned_text
    return text  # Return non-string values as-is



import re

def remove_special_characters(text):
    if isinstance(text, str):  # Ensure the input is a string
        # Remove special characters but keep accented characters (e.g., ê, ù, é)
        text = re.sub(r'[^A-Za-z0-9À-ÿ\s]', '', text)  # Allows accented characters (À-ÿ)
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    return text  # Return non-string values as-is

df_cleaned = df_cleaned.applymap(
    lambda text: remove_special_characters(text) if isinstance(text, str) else text
)

# Display the first few rows of the DataFrame after stopwords removal
# print(df_cleaned.head())




# Save the updated data
df_cleaned.to_csv('cleaned_no_stopwords.csv', index=False)
print("Stopwords removed and new file saved as cleaned_no_stopwords.csv")



