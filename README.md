# 🍽️ DS-Project – Data Science Pipeline (Food Dataset Analysis & ML Models)

This repository contains a complete **end-to-end Data Science pipeline** including data cleaning, preprocessing, feature engineering, encoding, analysis, and multiple machine learning models for prediction.

---

## 📌 Project Overview

The project works on a dataset of dishes and performs:

- Data Cleaning & Preprocessing
- Stopwords Removal (NLP Processing)
- Exploratory Data Analysis (EDA)
- Feature Encoding (Label, One-Hot, Frequency Encoding)
- Machine Learning Model Training
- Model Evaluation (Accuracy, Confusion Matrix, Classification Report)
- XGBoost & Naive Bayes Implementation

---

## 📂 Project Structure

DS-Project/
│
├── Analysis.py # Word frequency + EDA visualization
├── Cleanning.py # Data cleaning (missing values, duplicates, columns)
├── Stopwords.py # Stopwords removal + text preprocessing
├── Encoding.py # Feature encoding (Label, One-hot, Frequency)
├── encodding2.py # Improved encoding pipeline
│
├── Testing_Training.py # Logistic Regression model
├── testing-training.py # Random Forest + K-Fold validation
├── training-testing.py # Naive Bayes model
├── testing-training3.py # XGBoost (DMatrix version)
├── testing-training4.py # XGBoost (Classifier version)
│
├── ds-dataset.csv # Raw dataset
├── new_file.csv # Cleaned dataset
├── cleaned_no_stopwords.csv # NLP processed dataset
├── analyzed_dataset.csv # EDA output dataset
├── encoded_dataset.csv # Encoded dataset (ML ready)
├── encoded2_dataset.csv # Improved encoding version
│
├── xgboost_model.json # Saved trained model
└── Analysis Figure.png # Visualization output


---

## ⚙️ Technologies Used

- Python 🐍
- Pandas
- NumPy
- Scikit-Learn
- XGBoost
- Matplotlib
- NLTK (Natural Language Processing)

---

## 🔄 Data Pipeline Workflow
Raw Dataset
↓
Cleaning (Cleanning.py)
↓
Stopwords Removal (Stopwords.py)
↓
EDA Analysis (Analysis.py)
↓
Encoding (Encoding.py / encodding2.py)
↓
Model Training
├── Logistic Regression
├── Random Forest
├── Naive Bayes
└── XGBoost
↓
Evaluation (Accuracy, Confusion Matrix)

---

## 🤖 Machine Learning Models Used

### 1. Logistic Regression
- Simple baseline classifier
- Used for performance comparison

### 2. Random Forest Classifier
- Ensemble learning model
- Includes K-Fold validation

### 3. Naive Bayes (GaussianNB)
- Works well for probabilistic classification

### 4. XGBoost Classifier
- High-performance gradient boosting model
- Best accuracy among tested models

---

## 📊 Evaluation Metrics

- Accuracy Score
- Confusion Matrix
- Classification Report
- Cross Validation (K-Fold)

---

## 📈 Example Output
Accuracy: 87.45%
Confusion Matrix:
[[45 3]
[ 6 46]]

---

## 🚀 How to Run

1. Clone repository
```bash
git clone https://github.com/your-username/DS-Project.git
```

2. Install dependencies
```bash
pip install pandas numpy scikit-learn matplotlib nltk xgboost
```
3. Run pipeline step by step:
```bash
python Cleanning.py
python Stopwords.py
python Analysis.py
python Encoding.py
python Testing_Training.py
```

## 📌 Key Features

- Complete Data Science pipeline  
- NLP preprocessing included  
- Multiple ML models comparison  
- Feature encoding strategies  
- Model saving with XGBoost  

---

## 👩‍💻 Author

**Hafsa Rahman**  
Software Engineering Student  
Passionate about Data Science & Machine Learning  

---

## ⭐ Future Improvements

- Add Deep Learning models  
- Deploy model using Flask/Django  
- Create web dashboard for predictions  
