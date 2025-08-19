import logging
from src.fraud_predictor import FraudPredictor

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    predictor = FraudPredictor()
    predictor.process_and_train({'fraud': 'data/Fraud_Data.csv', 'ip': 'data/IpAddress_to_Country.csv'})