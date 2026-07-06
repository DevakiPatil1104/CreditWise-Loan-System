# LoanLens – ML-Based Loan Approval Predictor
A machine learning web application that predicts whether a loan application is likely to be approved based on an applicant's demographic, financial, and employment details. The project combines exploratory data analysis, feature engineering, model training, and deployment into a complete end-to-end ML pipeline.

---

## Project Overview
Financial institutions receive thousands of loan applications, making manual screening both time-consuming and inconsistent. This project demonstrates how machine learning can assist in evaluating loan eligibility by learning patterns from historical applicant data. The application allows users to enter applicant details through a simple web interface and instantly receive a loan approval prediction.

---

## Features
- Exploratory Data Analysis (EDA) with informative visualizations
- Data preprocessing and feature engineering
- Handling of missing values and categorical features
- Multiple machine learning models evaluated
- Hyperparameter tuning for improved performance
- Model serialization using Pickle
- Interactive Streamlit web application
- Real-time loan approval prediction

---

## Tech Stack

### Programming Language

- Python

### Libraries

- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Pickle

### Deployment

- Streamlit

---

## Project Workflow
1. Data Collection
2. Data Cleaning
3. Exploratory Data Analysis (EDA)
4. Feature Engineering
5. Data Preprocessing
6. Model Training
7. Model Evaluation
8. Hyperparameter Tuning
10. Model Serialization
11. Streamlit Deployment

---

## Model Performance
Different machine learning algorithms were trained and evaluated before selecting the final model based on overall performance.

### Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

> **Best Performing Model:** *Gaussian Navie Bayes Supervised ML Model*

---

## Application Preview


---

## Installation

### Clone the repository

```bash
git clone https://github.com/DevakiPatil1104/loanlens-ml.git
```

### Navigate to the project directory

```bash
cd miniProject_Loan_Approval_System
```

### Run the Streamlit application

```bash
streamlit run app.py
```

---

## Future Improvements

- Deploy the application on Streamlit Cloud or Render
- Integrate Explainable AI (SHAP/LIME)
- Display loan approval probability score
- Store prediction history in a database
- User authentication and role-based access
- Continuous model retraining with new data

---

## Learning Outcomes

Through this project, I gained practical experience in:

- Building an end-to-end machine learning pipeline
- Data preprocessing and feature engineering
- Exploratory Data Analysis using visualizations
- Model training, evaluation, and selection
- Hyperparameter tuning
- Building interactive ML applications with Streamlit
- Deploying machine learning models for real-time predictions
