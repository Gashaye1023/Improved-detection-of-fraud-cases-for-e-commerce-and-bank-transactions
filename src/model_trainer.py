
## src/model_trainer.py
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import shap
import joblib
import logging

class ModelTrainer:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.models = {
            'logistic': LogisticRegression(random_state=42),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42)
        }
        self.explainer = None
        
    def train_models(self, X_train, y_train):
        """Train multiple models."""
        results = {}
        for name, model in self.models.items():
            model.fit(X_train, y_train)
            results[name] = self.evaluate(model, X_train, y_train)
        self.explainer = shap.TreeExplainer(self.models['random_forest'])
        return results
    
    def evaluate(self, model, X_test, y_test):
        """Evaluate model performance."""
        y_pred = model.predict(X_test)
        return {
            'report': classification_report(y_test, y_pred, output_dict=True),
            'auc_pr': roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        }
    
    def get_shap_values(self, X):
        """Get SHAP values for explainability."""
        return self.explainer.shap_values(X)