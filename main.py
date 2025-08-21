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
import joblib
import yaml
from typing import Dict, Tuple, Optional
from pathlib import Path

# Configure logging with file handler
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/pipeline.log')
    ]
)
logger = logging.getLogger(__name__)

class DataLoader:
    """Handles loading and cleaning of fraud and IP mapping data."""
    
    def __init__(self, fraud_path: str, ip_path: str):
        self.fraud_path = fraud_path
        self.ip_path = ip_path
        self.fraud_df: Optional[pd.DataFrame] = None
        self.ip_df: Optional[pd.DataFrame] = None

    def load_and_clean_fraud_data(self) -> pd.DataFrame:
        """Load and clean fraud transaction data."""
        try:
            self.fraud_df = pd.read_csv(self.fraud_path)
            self.fraud_df.drop_duplicates(inplace=True)
            num_cols = self.fraud_df.select_dtypes(include='number').columns
            self.fraud_df[num_cols] = self.fraud_df[num_cols].fillna(self.fraud_df[num_cols].median())
            logger.info("Fraud data loaded and cleaned successfully.")
            return self.fraud_df
        except FileNotFoundError:
            logger.error(f"Fraud data file not found at {self.fraud_path}")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Error loading fraud data: {e}")
            sys.exit(1)

    def load_ip_mapping(self) -> pd.DataFrame:
        """Load IP address to country mapping data."""
        try:
            self.ip_df = pd.read_csv(self.ip_path)
            logger.info("IP mapping data loaded successfully.")
            return self.ip_df
        except FileNotFoundError:
            logger.error(f"IP mapping file not found at {self.ip_path}")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Error loading IP mapping data: {e}")
            sys.exit(1)

class DataProcessor:
    """Handles data merging, feature engineering, and preprocessing."""
    
    def __init__(self, fraud_df: pd.DataFrame, ip_df: pd.DataFrame):
        self.fraud_df = fraud_df
        self.ip_df = ip_df
        self.processed_df: Optional[pd.DataFrame] = None

    def ip_to_int(self, ip: any) -> int:
        """Convert IP address (as float) to integer."""
        try:
            return int(float(ip))
        except (ValueError, AttributeError) as e:
            logger.warning(f"Invalid IP address format: {ip}, error: {e}")
            return 0

    def merge_geolocation(self) -> pd.DataFrame:
        """Merge fraud data with geolocation information."""
        try:
            self.fraud_df['ip_int'] = self.fraud_df['ip_address'].apply(self.ip_to_int)
            self.fraud_df['country'] = 'Unknown'
            for index, row in self.ip_df.iterrows():
                mask = (self.fraud_df['ip_int'] >= row['lower_bound_ip_address']) & \
                       (self.fraud_df['ip_int'] <= row['upper_bound_ip_address'])
                self.fraud_df.loc[mask, 'country'] = row['country']
            logger.info("Geolocation merged successfully.")
            return self.fraud_df
        except Exception as e:
            logger.error(f"Error merging geolocation: {e}")
            sys.exit(1)

    def feature_engineering(self) -> pd.DataFrame:
        """Perform feature engineering on timestamp data."""
        try:
            self.fraud_df['signup_time'] = pd.to_datetime(self.fraud_df['signup_time'])
            self.fraud_df['purchase_time'] = pd.to_datetime(self.fraud_df['purchase_time'])
            self.fraud_df['timesincesignup'] = (self.fraud_df['purchase_time'] - self.fraud_df['signup_time']).dt.total_seconds()
            self.fraud_df['hourof_day'] = self.fraud_df['purchase_time'].dt.hour
            self.fraud_df['day_of_week'] = self.fraud_df['purchase_time'].dt.dayofweek
            logger.info("Feature engineering completed.")
            return self.fraud_df
        except Exception as e:
            logger.error(f"Error in feature engineering: {e}")
            sys.exit(1)

    def preprocess_data(self, output_path: str) -> pd.DataFrame:
        """Preprocess data and save to CSV."""
        try:
            self.processed_df = pd.get_dummies(self.fraud_df, columns=['browser', 'source', 'sex', 'country'], drop_first=True)
            self.processed_df.drop(columns=['user_id', 'device_id', 'ip_address', 'signup_time', 'purchase_time', 'ip_int'], inplace=True)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            self.processed_df.to_csv(output_path, index=False)
            logger.info(f"Processed data saved to {output_path}")
            return self.processed_df
        except Exception as e:
            logger.error(f"Error in data preprocessing: {e}")
            sys.exit(1)

class ModelTrainer:
    """Handles model training, evaluation, and saving."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.lr = LogisticRegression(max_iter=1000, random_state=42)
        self.rf = RandomForestClassifier(n_estimators=100, random_state=42)

    def balance_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Balance dataset using SMOTE."""
        try:
            sm = SMOTE(random_state=42)
            X_bal, y_bal = sm.fit_resample(X, y)
            logger.info("Data balanced using SMOTE.")
            return X_bal, y_bal
        except Exception as e:
            logger.error(f"Error balancing data: {e}")
            sys.exit(1)

    def train_and_evaluate(self, X: pd.DataFrame, y: pd.Series, model_dir: str) -> Tuple[LogisticRegression, RandomForestClassifier, float, float]:
        """Train and evaluate models, save them to disk."""
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            X_train_bal, y_train_bal = self.balance_data(X_train_scaled, y_train)
            
            # Train Logistic Regression
            self.lr.fit(X_train_bal, y_train_bal)
            y_pred_lr = self.lr.predict(X_test_scaled)
            
            # Train Random Forest
            self.rf.fit(X_train_bal, y_train_bal)
            y_pred_rf = self.rf.predict(X_test_scaled)
            
            # Save models
            os.makedirs(model_dir, exist_ok=True)
            lr_path = os.path.join(model_dir, 'logistic_regression_model.pkl')
            rf_path = os.path.join(model_dir, 'random_forest_model.pkl')
            best_model_path = os.path.join(model_dir, 'best_fraud_model.pkl')
            joblib.dump(self.lr, lr_path)
            joblib.dump(self.rf, rf_path)
            logger.info(f"Logistic Regression model saved to {lr_path}")
            logger.info(f"Random Forest model saved to {rf_path}")
            
            # Evaluate models
            f1_lr = f1_score(y_test, y_pred_lr)
            f1_rf = f1_score(y_test, y_pred_rf)
            conf_mat_lr = confusion_matrix(y_test, y_pred_lr)
            conf_mat_rf = confusion_matrix(y_test, y_pred_rf)
            precision_lr, recall_lr, _ = precision_recall_curve(y_test, self.lr.predict_proba(X_test_scaled)[:, 1])
            pr_auc_lr = auc(recall_lr, precision_lr)
            
            logger.info(f"Logistic Regression F1-Score: {f1_lr}, PR-AUC: {pr_auc_lr}")
            logger.info(f"Random Forest F1-Score: {f1_rf}")
            logger.info(f"Logistic Regression Confusion Matrix:\n{conf_mat_lr}")
            logger.info(f"Random Forest Confusion Matrix:\n{conf_mat_rf}")
            
            # Save best model
            best_model = self.lr if f1_lr > f1_rf else self.rf
            joblib.dump(best_model, best_model_path)
            logger.info(f"Best model ({'Logistic Regression' if f1_lr > f1_rf else 'Random Forest'}) saved to {best_model_path}")
            
            return self.lr, self.rf, f1_lr, f1_rf
        except Exception as e:
            logger.error(f"Error in training and evaluation: {e}")
            sys.exit(1)

def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        logger.info(f"Configuration loaded from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Load configuration
    config_path = 'config.yaml'
    config = load_config(config_path)
    
    # Initialize components
    data_loader = DataLoader(config['fraud_path'], config['ip_path'])
    fraud_df = data_loader.load_and_clean_fraud_data()
    ip_df = data_loader.load_ip_mapping()
    
    # Process data
    processor = DataProcessor(fraud_df, ip_df)
    fraud_df = processor.merge_geolocation()
    fraud_df = processor.feature_engineering()
    processed_df = processor.preprocess_data(config['processed_data_path'])
    
    # Train and evaluate models
    trainer = ModelTrainer()
    lr, rf, f1_lr, f1_rf = trainer.train_and_evaluate(processed_df.drop(columns=['class']), processed_df['class'], config['model_dir'])
    
    logger.info(f"Best model (based on F1-score): {'Logistic Regression' if f1_lr > f1_rf else 'Random Forest'}")