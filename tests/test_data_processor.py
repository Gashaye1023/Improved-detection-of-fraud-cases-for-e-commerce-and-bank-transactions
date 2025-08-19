
## tests/test_data_processor.py
import pytest
from src.data_processor import DataProcessor

def test_load_and_clean():
    processor = DataProcessor()
    data = processor.load_and_clean({'test': 'data/Fraud_Data.csv'})
    assert not data['test'].isna().any().any()
    assert len(data['test']) > 0

