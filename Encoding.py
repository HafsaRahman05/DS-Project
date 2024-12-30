import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load the dataset
file_path = 'analyzed_dataset.csv'  # Replace with your file path
df = pd.read_csv(file_path)

# Identify categorical columns
categorical_columns = df.select_dtypes(include=['object']).columns

# Step 1: Frequency Encoding for high-cardinality columns
high_cardinality_columns = ['Dish Name', 'Region/Origin', 'Ingredients', 'Cultural Significance']
for col in high_cardinality_columns:
    frequency_map = df[col].value_counts().to_dict()
    df[col] = df[col].map(frequency_map)

# Step 2: Label Encoding for moderate-cardinality columns
label_encoders = {}
moderate_cardinality_columns = [col for col in categorical_columns if col not in high_cardinality_columns]
for col in moderate_cardinality_columns:
    if df[col].nunique() > 1000:  # If cardinality is high for one-hot encoding
        label_encoders[col] = LabelEncoder()
        df[col] = label_encoders[col].fit_transform(df[col].astype(str))

# Step 3: One-Hot Encoding for low-cardinality columns
low_cardinality_columns = [col for col in moderate_cardinality_columns if df[col].nunique() <= 1000]
if low_cardinality_columns:
    df = pd.get_dummies(df, columns=low_cardinality_columns, drop_first=True)

# Save the encoded dataset
df.to_csv('encoded_dataset.csv', index=False)
print("Dataset encoded and saved as 'encoded_dataset.csv'")
