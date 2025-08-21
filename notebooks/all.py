import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, confusion_matrix, precision_recall_curve, auc
from imblearn.over_sampling import SMOTE
import logging
import sys
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FraudDetectionPipeline:
    """A class to manage the end-to-end fraud detection pipeline for e-commerce transactions."""
    
    def __init__(self, config):
        """Initialize with configuration dictionary containing file paths."""
        self.config = config
        self.fraud_df = None
        self.ip_df = None
        self.processed_df = None

    def load_and_clean_fraud_data(self):
        """Load and clean the fraud transaction data."""
        try:
            self.fraud_df = pd.read_csv(self.config['fraud_path'])
            self.fraud_df.drop_duplicates(inplace=True)
            num_cols = self.fraud_df.select_dtypes(include='number').columns
            self.fraud_df[num_cols] = self.fraud_df[num_cols].fillna(self.fraud_df[num_cols].median())
            logger.info("Fraud data loaded and cleaned successfully.")
        except FileNotFoundError:
            logger.error(f"Fraud data file not found at {self.config['fraud_path']}")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Error loading fraud data: {e}")
            sys.exit(1)
        return self.fraud_df

    def load_ip_mapping(self):
        """Load the IP address to country mapping data."""
        try:
            self.ip_df = pd.read_csv(self.config['ip_path'])
            logger.info("IP mapping data loaded successfully.")
        except FileNotFoundError:
            logger.error(f"IP mapping file not found at {self.config['ip_path']}")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Error loading IP mapping data: {e}")
            sys.exit(1)
        return self.ip_df

    def ip_to_int(self, ip):
        """Convert IP address (as float) to integer."""
        try:
            return int(float(ip))
        except (ValueError, AttributeError) as e:
            logger.warning(f"Invalid IP address format: {ip}, error: {e}")
            return 0

    def merge_geolocation(self):
        """Merge fraud data with geolocation information using range check."""
        try:
            self.fraud_df['ip_int'] = self.fraud_df['ip_address'].apply(self.ip_to_int)
            self.fraud_df['country'] = 'Unknown'
            for index, row in self.ip_df.iterrows():
                mask = (self.fraud_df['ip_int'] >= row['lower_bound_ip_address']) & (self.fraud_df['ip_int'] <= row['upper_bound_ip_address'])
                self.fraud_df.loc[mask, 'country'] = row['country']
            logger.info("Geolocation merged successfully.")
        except Exception as e:
            logger.error(f"Error merging geolocation: {e}")
            return None  # Allow caller to handle failure
        return self.fraud_df

    def feature_engineering(self):
        """Perform feature engineering on timestamp data."""
        try:
            self.fraud_df['signup_time'] = pd.to_datetime(self.fraud_df['signup_time'])
            self.fraud_df['purchase_time'] = pd.to_datetime(self.fraud_df['purchase_time'])
            self.fraud_df['timesincesignup'] = (self.fraud_df['purchase_time'] - self.fraud_df['signup_time']).dt.total_seconds()
            self.fraud_df['hourof_day'] = self.fraud_df['purchase_time'].dt.hour
            self.fraud_df['day_of_week'] = self.fraud_df['purchase_time'].dt.dayofweek
            logger.info("Feature engineering completed.")
        except Exception as e:
            logger.error(f"Error in feature engineering: {e}")
            sys.exit(1)
        return self.fraud_df

    def preprocess_data(self):
        """Preprocess data by encoding categorical variables and dropping unnecessary columns."""
        try:
            self.processed_df = pd.get_dummies(self.fraud_df, columns=['browser', 'source', 'sex', 'country'], drop_first=True)
            self.processed_df.drop(columns=['user_id', 'device_id', 'ip_address', 'signup_time', 'purchase_time', 'ip_int'], inplace=True)
            logger.info("Data preprocessing completed.")
        except Exception as e:
            logger.error(f"Error in data preprocessing: {e}")
            sys.exit(1)
        return self.processed_df

    def balance_data(self, X, y):
        """Balance the dataset using SMOTE."""
        try:
            sm = SMOTE(random_state=42)
            X_bal, y_bal = sm.fit_resample(X, y)
            logger.info("Data balanced using SMOTE.")
            return X_bal, y_bal
        except Exception as e:
            logger.error(f"Error balancing data: {e}")
            sys.exit(1)

    def train_and_evaluate(self):
        """Train and evaluate Logistic Regression and Random Forest models."""
        try:
            X = self.processed_df.drop(columns=['class'])
            y = self.processed_df['class']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            X_train_bal, y_train_bal = self.balance_data(X_train_scaled, y_train)
            
            # Logistic Regression
            lr = LogisticRegression(max_iter=1000, random_state=42)
            lr.fit(X_train_bal, y_train_bal)
            y_pred_lr = lr.predict(X_test_scaled)
            
            # Random Forest
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(X_train_bal, y_train_bal)
            y_pred_rf = rf.predict(X_test_scaled)
            
            # Additional evaluation metrics
            f1_lr = f1_score(y_test, y_pred_lr)
            f1_rf = f1_score(y_test, y_pred_rf)
            conf_mat_lr = confusion_matrix(y_test, y_pred_lr)
            conf_mat_rf = confusion_matrix(y_test, y_pred_rf)
            precision_lr, recall_lr, _ = precision_recall_curve(y_test, lr.predict_proba(X_test_scaled)[:, 1])
            pr_auc_lr = auc(recall_lr, precision_lr)
            
            logger.info(f"Logistic Regression F1-Score: {f1_lr}, PR-AUC: {pr_auc_lr}")
            logger.info(f"Random Forest F1-Score: {f1_rf}")
            logger.info(f"Logistic Regression Confusion Matrix:\n{conf_mat_lr}")
            logger.info(f"Random Forest Confusion Matrix:\n{conf_mat_rf}")
            
            return lr, rf, f1_lr, f1_rf
        except Exception as e:
            logger.error(f"Error in training and evaluation: {e}")
            sys.exit(1)

if __name__ == "__main__":
    # Configuration
    config = {
        'fraud_path': os.path.join('data', 'Fraud_Data.csv'),
        'ip_path': os.path.join('data', 'IpAddress_to_Country.csv')
    }
    
    # Initialize and run pipeline
    pipeline = FraudDetectionPipeline(config)
    pipeline.load_and_clean_fraud_data()
    pipeline.load_ip_mapping()
    pipeline.merge_geolocation()
    pipeline.feature_engineering()
    pipeline.preprocess_data()
    models = pipeline.train_and_evaluate()
    
    # Log best model based on F1-score
    lr, rf, f1_lr, f1_rf = models
    best_model = lr if f1_lr > f1_rf else rf
    logger.info(f"Best model (based on F1-score): {'Logistic Regression' if f1_lr > f1_rf else 'Random Forest'}")
    
    # Save models
    import joblib
    joblib.dump(best_model, os.path.join('src', 'best_fraud_model.pkl'))
    logger.info("Best model saved to src/best_fraud_model.pkl")