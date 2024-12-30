import pandas as pd
import xgboost as xgb 
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error

# Load dataset
file_path = 'encoded_dataset.csv'  # Replace with your file path
df = pd.read_csv(file_path)

# # Specify the task: 'classification' or 'regression'
# task_type = 'classification'  # Change to 'regression' for regression tasks

# Features and target
# target_column = 'target_column'  # Replace with your target column name
X = df.drop(columns=['Dish Name'])
y = df['Cultural Significance']

# Check if the labels are binary
print(y.unique())

# Convert labels to binary (0 and 1)
# Example: If your classes are 'Class1' and 'Class2', map them to 0 and 1
y = y.apply(lambda x: 1 if x == 'Class1' else 0)  # Adjust based on your dataset

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

params = {
    'objective': 'binary:logistic',  # For binary classification
    'eval_metric': 'logloss',        # Evaluation metric for binary classification
    'learning_rate': 0.1,
    'max_depth': 6
}

# Convert data to DMatrix (XGBoost's optimized data structure)
train_data = xgb.DMatrix(data=X_train, label=y_train)
test_data = xgb.DMatrix(data=X_test, label=y_test)

# Train the model
model = xgb.train(params, train_data, num_boost_round=100)

# Alternatively, if using XGBClassifier, use this:
# model = XGBClassifier(
#     objective='binary:logistic',
#     eval_metric='logloss',
#     learning_rate=0.1,
#     max_depth=6,
#     n_estimators=100
# )
# model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(test_data)

# If using XGBClassifier, use:
# y_pred = model.predict(X_test)

# Convert probabilities to binary (0 or 1)
y_pred_binary = (y_pred > 0.5).astype(int)  # For binary classification, use a threshold of 0.5

# Evaluate the model
from sklearn.metrics import accuracy_score, confusion_matrix

# Accuracy
accuracy = accuracy_score(y_test, y_pred_binary)
print(f"Accuracy: {accuracy}")
print(f'Accuracy: {accuracy * 100:.2f}%')

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_binary)
print("Confusion Matrix:")
print(cm)

# Save the trained model for future use
model.save_model('xgboost_model.json')
