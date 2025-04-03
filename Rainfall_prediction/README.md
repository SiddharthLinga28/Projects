# 🌧️ Rainfall Prediction Using Machine Learning

### 📌 Developed By:
- **Siddharth Linga** – Dept. of Computer Science, Texas A&M University - Corpus Christi  
  📧 slinga1@islander.tamucc.edu  

---

## 📝 Abstract

Rainfall prediction is crucial for planning in agriculture, flood prevention, and disaster management. This project uses **machine learning techniques** to predict whether it will rain the next day using historical Australian weather data. The system follows a complete ML lifecycle—data preprocessing, model training, evaluation, and deployment—using algorithms like **Logistic Regression, Decision Tree, Random Forest, XGBoost, and LightGBM**.

---

## 🎯 Objectives

- To preprocess and clean the Australian weather dataset.
- To build and compare multiple ML models for rainfall prediction.
- To evaluate each model using performance metrics (Accuracy, Precision, Recall, F1-score).
- To deploy a **Streamlit-based web app** for real-time rainfall prediction.

---

## 🧠 Methodology

### 🔍 Exploratory Data Analysis (EDA)
- Visualizing distributions
- Identifying correlations
- Dealing with missing values and outliers

### ⚙️ Data Preprocessing
- Handling missing values via imputation
- Encoding categorical variables (Label Encoding)
- Feature scaling using StandardScaler
- Removing unnecessary features

### 🤖 Model Training
- Algorithms used:
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - XGBoost
  - LightGBM
- Evaluation Metrics:
  - Accuracy
  - Precision
  - Recall
  - F1-score
- Model validation using train-test split (80:20)

### 🖥️ Web Application
- Built with **Streamlit**
- Allows users to input features (like temperature, humidity, pressure, etc.)
- Returns a prediction: **Will it rain tomorrow? Yes/No**

---

## 📊 Results

| Algorithm         | Accuracy | Precision | Recall | F1-score |
|------------------|----------|-----------|--------|----------|
| Random Forest     | 93.43%   | 0.75      | 0.49   | 0.60     |
| LightGBM          | 93.43%   | 0.75      | 0.52   | 0.61     |
| Logistic Regression | 93.37% | 0.72      | 0.46   | 0.56     |
| Decision Tree     | 87.70%   | 0.51      | 0.52   | 0.52     |
| XGBoost           | 93.43%   | 0.97      | 0.03   | 0.05     |

🎯 **Best performing models**: LightGBM & Random Forest

---

## 🗃️ Dataset

- Source: [Kaggle Rainfall in Australia Dataset](https://www.kaggle.com/)
- Size: ~100,000 rows with 23 features
- Target variable: `RainTomorrow` (Yes/No)

---

## 🧪 Test Cases

| Test Case                            | Expected Result                          | Status |
|-------------------------------------|------------------------------------------|--------|
| Large dataset handling              | Model handles data efficiently           | ✅     |
| Null value imputation               | Handled via mode or mean                 | ✅     |
| Categorical feature encoding        | Encoded using LabelEncoder               | ✅     |
| String values in model input        | Converted successfully                   | ✅     |
| Model accuracy                      | Achieved ~93% accuracy                   | ✅     |

---

## 📂 Folder Structure

```bash
📁 rainfall-prediction-ml/
├── data/                     # Sample dataset
├── models/                   # Saved ML models
├── notebooks/                # Jupyter/EDA notebooks
├── src/                      # Model training, preprocessing scripts
├── webapp/                   # Streamlit application
├── results/                  # Confusion matrices and plots
├── README.md                 # This file
└── requirements.txt          # Python dependencies
