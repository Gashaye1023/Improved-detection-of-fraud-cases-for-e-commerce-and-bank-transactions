## src/fraud_predictor.py
from .data_processor import DataProcessor
from .model_trainer import ModelTrainer
import logging

class FraudPredictor:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.processor = DataProcessor()
        self.trainer = ModelTrainer()
        
    def process_and_train(self, file_paths: dict):
        """Process data and train models."""
        data = self.processor.load_and_clean(file_paths)
        fraud_df = self.processor.merge_geolocation(data['fraud'], data['ip'])
        fraud_df = self.processor.feature_engineering(fraud_df)
        X_train, X_test, y_train, y_test = self.processor.preprocess(fraud_df, 'class')
        results = self.trainer.train_models(X_train, y_train)
        self.logger.info(f"Training results: {results}")
        return X_test, y_test, results
    
    def predict(self, df):
        """Predict fraud probabilities."""
        X, _ = self.processor.preprocess(df, 'class')[0:2]
        return self.trainer.models['random_forest'].predict_proba(X)[:, 1]