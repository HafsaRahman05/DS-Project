import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Load your dataset
file_path = 'encoded_dataset.csv'  # Replace with your actual dataset path
df = pd.read_csv(file_path)

# Assuming 'target_column' is the target variable and other columns are features
X = df.drop(columns=['Dish Name'])
y = df['Cultural Significance']

# Check if the labels are binary (0 or 1)
print(y.unique())

# Convert labels to binary (if necessary)
y = y.apply(lambda x: 1 if x == 'Class1' else 0)  # Adjust this for your labels

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the XGBClassifier with the chosen parameters
model = XGBClassifier(
    objective='binary:logistic',  # Binary classification task
    eval_metric='logloss',        # Evaluation metric for binary classification
    learning_rate=0.1,
    max_depth=6,
    n_estimators=100
)

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print(f'Accuracy: {accuracy * 100:.2f}%')

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)
