
## tests/test_model_trainer.py

import pytest
from src.model_trainer import ModelTrainer

def test_train_models():
    trainer = ModelTrainer()
    X = [[1, 2], [3, 4], [5, 6]]
    y = [0, 1, 0]
    results = trainer.train_models(X, y)
    assert all(model in results for model in ['logistic', 'random_forest'])