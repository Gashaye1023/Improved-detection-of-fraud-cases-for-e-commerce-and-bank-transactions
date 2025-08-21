## src/data_processor.py
import pandas as pd
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import logging

class DataProcessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.logger = logging.getLogger(__name__)
        
    def load_and_clean(self, file_paths: dict) -> dict:
        """Load and clean datasets."""
        data = {}
        for name, path in file_paths.items():
            df = pd.read_csv(path).drop_duplicates().dropna()
            data[name] = df
        self.logger.info("Data loaded and cleaned")
        return data
    
    def merge_geolocation(self, fraud_df: pd.DataFrame, ip_df: pd.DataFrame) -> pd.DataFrame:
        """Merge fraud data with IP to country mapping."""
        # Handle NaN values in the fraud DataFrame
        fraud_df['ip_address'] = fraud_df['ip_address'].fillna('0.0.0.0')
        fraud_df['ip_int'] = fraud_df['ip_address'].apply(ip_to_int)

        # Create integer columns in the IP DataFrame
        ip_df['lower_int'] = ip_df['lower_bound_ip_address'].apply(ip_to_int)
        ip_df['upper_int'] = ip_df['upper_bound_ip_address'].apply(ip_to_int)

        # Debugging: Check the columns and sample data
        self.logger.debug("Columns in IP DataFrame: %s", ip_df.columns)
        self.logger.debug("Sample data in IP DataFrame:\n%s", ip_df.head())

        # Ensure required columns exist
        if 'lower_int' not in ip_df.columns or 'upper_int' not in ip_df.columns:
            self.logger.error("Required columns are missing in IP DataFrame")
            raise KeyError("Required columns are missing in IP DataFrame")

        # Perform the merge
        merged = pd.merge_asof(
            fraud_df.sort_values('ip_int'), 
            ip_df.sort_values('lower_int'),
            left_on='ip_int', right_on='lower_int',
            by='upper_int', direction='nearest'
        )
        
        return merged.drop(['ip_int', 'lower_int', 'upper_int'], axis=1)
    
    def feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer features for fraud detection."""
        df['purchase_time'] = pd.to_datetime(df['purchase_time'])
        df['signup_time'] = pd.to_datetime(df['signup_time'])
        df['hour_of_day'] = df['purchase_time'].dt.hour
        df['day_of_week'] = df['purchase_time'].dt.dayofweek
        df['time_since_signup'] = (df['purchase_time'] - df['signup_time']).dt.total_seconds() / 3600
        return df
    
    def preprocess(self, df: pd.DataFrame, target_col: str) -> tuple:
        """Preprocess data and handle imbalance."""
        X = pd.get_dummies(df.drop(target_col, axis=1))
        y = df[target_col]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
        smote = SMOTE(random_state=42)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
        X_train_scaled = pd.DataFrame(self.scaler.fit_transform(X_train_res), columns=X_train_res.columns)
        X_test_scaled = pd.DataFrame(self.scaler.transform(X_test), columns=X_test.columns)
        return (X_train_scaled, X_test_scaled, y_train_res, y_test)

def ip_to_int(ip: str) -> int:
    """Convert IP address to integer."""
    if isinstance(ip, str):  # Check if ip is a string
        parts = ip.split('.')
        if len(parts) == 4:  # Ensure it has 4 parts
            return sum(int(x) << (24 - 8 * i) for i, x in enumerate(parts))
    return 0  # Return 0 for invalid input