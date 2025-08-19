# Improved-detection-of-fraud-cases-for-e-commerce-and-bank-transactions
# Fraud Detection Project

## Overview

This project aims to improve the detection of fraud cases in e-commerce and banking transactions. It uses advanced machine learning techniques to identify fraudulent activities, enhancing transaction security and customer trust.

## Data Sources

- **Fraud_Data.csv**: Contains e-commerce transactions.
- **IpAddress_to_Country.csv**: Maps IP addresses to countries.
- **creditcard.csv**: Bank transaction data for fraud detection.

## Project Structure

- `data/`: Contains raw datasets.
- `notebooks/`: Jupyter notebooks for analysis and model training.
- `src/`: Source code for data preprocessing, feature engineering, model selection, and explainability.
## Features
- Data preprocessing and geolocation merging
- Feature engineering (time-based features, transaction velocity)
- Logistic Regression and Random Forest models
- SHAP explainability
- Streamlit dashboard
- Unit tests and CI/CD
## Setup Instructions

1. Clone the repository:
    git clone https://github.com/Gashaye1023/Improved-detection-of-fraud-cases-for-e-commerce-and-bank-transactions.git
    cd Improved-detection-of-fraud-cases-for-e-commerce-and-bank-transactions
2. Install required packages:
    pip install -r requirements.txt
3. Run the data preprocessing script:
    python src/data_preprocessing.py
4. Train models using the notebook:
    - Open `notebooks/model_training.ipynb` and run the cells.
# Learning Outcomes
- Clean and preprocess complex datasets.
- Handle class imbalance effectively.
- Train and evaluate models using appropriate metrics.
- Interpret model predictions using SHAP.
# DATASET columns
- Fraud_Data.csv
- Includes e-commerce transaction data 
    user_id: A unique identifier for the user who made the transaction.
    signup_time: The timestamp when the user signed up.
    purchase_time: The timestamp when the purchase was made.
    purchase_value: The value of the purchase in dollars.
    device_id: A unique identifier for the device used to make the transaction.
    source: The source through which the user came to the site (e.g., SEO, Ads).
    browser: The browser used to make the transaction (e.g., Chrome, Safari).
    sex: The gender of the user (M for male, F for female).
    age: The age of the user.
    ip_address: The IP address from which the transaction was made.
    class: The target variable where 1 indicates a fraudulent transaction and 0 indicates a non-fraudulent transaction.
    Critical Challenge: Class Imbalance. This dataset is highly imbalanced, with far fewer fraudulent transactions than legitimate ones. This will significantly influence your choice of evaluation metrics and modeling techniques.
    
# IpAddress_to_Country.csv
# Maps IP addresses to countries

    lower_bound_ip_address: The lower bound of the IP address range.
    upper_bound_ip_address: The upper bound of the IP address range.
    country: The country corresponding to the IP address range.

 #   creditcard.csv
# Contains bank transaction data specifically curated for fraud detection analysis. 

    Time: The number of seconds elapsed between this transaction and the first transaction in the dataset.
    V1 to V28: These are anonymized features resulting from a PCA transformation. Their exact nature is not disclosed for privacy reasons, but they represent the underlying patterns in the data.
    Amount: The transaction amount in dollars.
    Class: The target variable where 1 indicates a fraudulent transaction and 0 indicates a non-fraudulent transaction.
    Critical Challenge: Class Imbalance. Like the e-commerce data, this dataset is extremely imbalanced, which is typical for fraud detection problems.