import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, confusion_matrix
from imblearn.over_sampling import SMOTE

def load_and_clean_fraud_data(fraud_path):
    # Read e-commerce transactions
    df = pd.read_csv(fraud_path)
    df.drop_duplicates(inplace=True)
    # Impute numeric missing values with median
    num_cols = df.select_dtypes(include='number').columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    return df

def load_ip_mapping(ip_path):
    # Read IP mapping data
    ip_df = pd.read_csv(ip_path)
    return ip_df

def ip_to_int(ip_str):
    # Simple conversion of IP address to integer
    parts = ip_str.split('.')
    return sum(int(part) << (8*(3-i)) for i, part in enumerate(parts))

def merge_geolocation(fraud_df, ip_df):
    # Convert ipaddress column to integer
    fraud_df['ip_int'] = fraud_df['ipaddress'].apply(ip_to_int)
    # Merge to fetch country info for each transaction (approximate join)
    def find_country(ip_val, ip_mapping):
        row = ip_mapping[(ip_mapping['lowerboundipaddress'] <= ip_val) & (ip_mapping['upperboundipaddress'] >= ip_val)]
        return row['country'].values[0] if not row.empty else 'Unknown'
    fraud_df['country'] = fraud_df['ip_int'].apply(lambda x: find_country(x, ip_df))
    return fraud_df

def feature_engineering(df):
    # Convert timestamps to datetime
    df['signuptime'] = pd.to_datetime(df['signuptime'])
    df['purchasetime'] = pd.to_datetime(df['purchasetime'])
    # Create timesincesignup in seconds
    df['timesincesignup'] = (df['purchasetime'] - df['signuptime']).dt.total_seconds()
    # Extract hour of purchase and day of week
    df['hourof_day'] = df['purchasetime'].dt.hour
    df['day_of_week'] = df['purchasetime'].dt.dayofweek
    return df

def preprocess_data(df):
    # One-Hot Encode categorical columns: browser, source, sex, country
    df_encoded = pd.get_dummies(df, columns=['browser', 'source', 'sex', 'country'], drop_first=True)
    # Drop columns not needed for modeling
    df_encoded.drop(columns=['userid', 'deviceid', 'ipaddress', 'signuptime', 'purchasetime'], inplace=True)
    return df_encoded

def balance_data(X, y):
    sm = SMOTE(random_state=42)
    return sm.fit_resample(X, y)

def train_and_evaluate(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Handle class imbalance only on training data
    X_train_bal, y_train_bal = balance_data(X_train_scaled, y_train)
    
    # Logistic Regression
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train_bal, y_train_bal)
    y_pred_lr = lr.predict(X_test_scaled)
    
    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train_bal, y_train_bal)
    y_pred_rf = rf.predict(X_test_scaled)
    
    print("Logistic Regression F1-Score:", f1_score(y_test, y_pred_lr))
    print("Random Forest F1-Score:", f1_score(y_test, y_pred_rf))
    print("Logistic Regression Confusion Matrix:\n", confusion_matrix(y_test, y_pred_lr))
    print("Random Forest Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))

if __name__ == "__main__":
    # File paths (update paths as necessary)
    fraud_file = 'FraudData.csv'
    ip_file = 'IpAddresstoCountry.csv'
    
    # Load and clean datasets
    fraud_df = load_and_clean_fraud_data(fraud_file)
    ip_df = load_ip_mapping(ip_file)
    
    # Merge to get geolocation features
    fraud_df = merge_geolocation(fraud_df, ip_df)
    
    # Feature engineering for timestamps
    fraud_df = feature_engineering(fraud_df)
    
    # Preprocess data (encoding, dropping unneeded columns)
    processed_df = preprocess_data(fraud_df)
    
    # Separate features and target ('Class')
    X = processed_df.drop(columns=['Class'])
    y = processed_df['Class']
    
    # Train models and evaluate
    train_and_evaluate(X, y)