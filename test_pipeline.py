import pytest
import pandas as pd
import numpy as np
from main import DataLoader, DataProcessor, ModelTrainer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import os
import yaml

# Load configuration
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    fraud_data = pd.DataFrame({
        'user_id': [1, 2],
        'signup_time': ['2023-01-01 10:00:00', '2023-01-02 12:00:00'],
        'purchase_time': ['2023-01-01 10:30:00', '2023-01-02 12:30:00'],
        'ip_address': ['192.168.1.1', '192.168.1.2'],
        'device_id': ['device1', 'device2'],
        'source': ['SEO', 'Direct'],
        'browser': ['Chrome', 'Firefox'],
        'sex': ['M', 'F'],
        'class': [0, 1]
    })
    ip_data = pd.DataFrame({
        'lower_bound_ip_address': [3232235776, 3232235777],
        'upper_bound_ip_address': [3232235776, 3232235777],
        'country': ['US', 'CA']
    })
    return fraud_data, ip_data

def test_load_and_clean_fraud_data(sample_data):
    """Test loading and cleaning fraud data."""
    fraud_data, _ = sample_data
    fraud_data.to_csv('test_fraud.csv', index=False)
    loader = DataLoader('test_fraud.csv', config['ip_path'])
    df = loader.load_and_clean_fraud_data()
    assert not df.empty
    assert df['user_id'].notnull().all()
    os.remove('test_fraud.csv')

def test_merge_geolocation(sample_data):
    """Test geolocation merging."""
    fraud_data, ip_data = sample_data
    processor = DataProcessor(fraud_data, ip_data)
    df = processor.merge_geolocation()
    assert 'country' in df.columns
    assert df['country'].iloc[0] == 'US'

def test_feature_engineering(sample_data):
    """Test feature engineering."""
    fraud_data, _ = sample_data
    processor = DataProcessor(fraud_data, None)
    df = processor.feature_engineering()
    assert 'timesincesignup' in df.columns
    assert 'hourof_day' in df.columns
    assert 'day_of_week' in df.columns

def test_preprocess_data(sample_data):
    """Test data preprocessing."""
    fraud_data, _ = sample_data
    processor = DataProcessor(fraud_data, None)
    processor.feature_engineering()
    df = processor.preprocess_data('test_processed.csv')
    assert 'class' in df.columns
    assert 'source_Direct' in df.columns
    os.remove('test_processed.csv')

def test_train_and_evaluate(sample_data):
    """Test model training and evaluation."""
    fraud_data, _ = sample_data
    processor = DataProcessor(fraud_data, None)
    processor.feature_engineering()
    df = processor.preprocess_data('test_processed.csv')
    trainer = ModelTrainer()
    lr, rf, f1_lr, f1_rf = trainer.train_and_evaluate(df.drop(columns=['class']), df['class'], 'test_models')
    assert isinstance(lr, LogisticRegression)
    assert isinstance(rf, RandomForestClassifier)
    assert os.path.exists('test_models/logistic_regression_model.pkl')
    assert os.path.exists('test_models/random_forest_model.pkl')
    assert os.path.exists('test_models/best_fraud_model.pkl')
    os.remove('test_processed.csv')
    os.remove('test_models/logistic_regression_model.pkl')
    os.remove('test_models/random_forest_model.pkl')
    os.remove('test_models/best_fraud_model.pkl')