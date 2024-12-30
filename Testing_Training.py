import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression  # You can replace this with any other model
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the encoded dataset
file_path = 'encoded_dataset.csv'  # Replace with your file path
df = pd.read_csv(file_path)

# Split the data into features (X) and target variable (y)
# Assuming 'target_column' is the column you want to predict
X = df.drop(columns=['Dish Name'])  # Replace 'target_column' with the actual column name
y = df['Cultural Significance']  # Replace 'target_column' with the actual column name
print(y.value_counts())  # This will show how many instances belong to each class
print(X.isnull().sum())  # Check for missing values in features
print(y.isnull().sum())  # Check for missing values in target variable

# Step 1: Split the dataset into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 2: Initialize and train the model
model = LogisticRegression()  # You can use any classifier or regressor
model.fit(X_train, y_train)

# Step 3: Predict on the test set
y_pred = model.predict(X_test)

# Step 4: Evaluate the model performance
# Accuracy Score
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Detailed Classification Report (for classification tasks)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix (for classification tasks)
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
