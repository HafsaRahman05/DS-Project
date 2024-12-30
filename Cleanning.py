import pandas as pd

# Load the CSV file into a pandas DataFrame
df = pd.read_csv('ds-dataset.csv', encoding='ISO-8859-1')  

# View dataset
print("Total Count Of Original Data:")
# pd.set_option('display.max_rows', None)  # Show all rows
# pd.set_option('display.max_columns', None)  # Show all column
print(df.shape)


# 1. Remove rows with missing values
# df_cleaned = df.dropna()  

# 2. Fill missing values with a specific value (e.g., 0 or 'Unknown')
df_filled = df.fillna(0)  # Replace NaN with 0 (or any value)

# 3. Remove duplicate rows
df_no_duplicates = df.drop_duplicates()
print("After Removing Duplicate Rows From Original Data:")
print(df.shape)

# 4. Remove extra column
df = df.drop(['Cooking Method', 'Serving Style', 'Spiciness Level', 'Dietary Information', 'Common Ingredients'], axis=1)
print("After Removing Extra Columns From Original Data:")
print(df.columns)
print(df.shape)


# 4. Filter rows based on a condition (e.g., keeping rows where 'age' > 30)
# df_filtered = df[df['age'] > 30]  # Replace 'age' with your column name and condition

# 5. Convert column types (e.g., changing a column to integer type)
# df['column_name'] = df['column_name'].astype(int)  # Replace 'column_name' with the column you want to change

# Save the cleaned data/ DataFrame to a new CSV file
df.to_csv('new_file.csv', index=False)  # index=False to exclude row indices from the saved file

# View the cleaned data
print("\nCleaned Data:")
print(df.to_csv)