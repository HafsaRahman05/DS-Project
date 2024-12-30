import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load the dataset
file_path = 'analyzed_dataset.csv'  # Replace with your file path
df = pd.read_csv(file_path)

# Identify categorical columns
categorical_columns = df.select_dtypes(include=['object']).columns

# Set thresholds
high_cardinality_threshold = 100  # Unique values > this threshold = High Cardinality
low_cardinality_threshold = 20    # Unique values <= this threshold = Low Cardinality

# Initialize encoders dictionary
label_encoders = {}

# Encoding Process
for col in categorical_columns:
    unique_count = df[col].nunique()
    print(f"Processing Column: {col} | Unique Values: {unique_count}")
    
    # Step 1: High-Cardinality Columns (Frequency Encoding)
    if unique_count > high_cardinality_threshold:
        print(f"Applying Frequency Encoding to '{col}'")
        frequency_map = df[col].value_counts().to_dict()
        df[col] = df[col].map(frequency_map)
    
    # Step 2: Moderate-Cardinality Columns (Label Encoding)
    elif high_cardinality_threshold >= unique_count > low_cardinality_threshold:
        print(f"Applying Label Encoding to '{col}'")
        label_encoders[col] = LabelEncoder()
        df[col] = label_encoders[col].fit_transform(df[col].astype(str))
    
    # Step 3: Low-Cardinality Columns (One-Hot Encoding)
    elif unique_count <= low_cardinality_threshold:
        print(f"Applying One-Hot Encoding to '{col}'")
        df = pd.get_dummies(df, columns=[col], drop_first=True)

# Save the encoded dataset
output_file = 'encoded2_dataset.csv'
df.to_csv(output_file, index=False)
print(f"Dataset encoded and saved as '{output_file}'")
