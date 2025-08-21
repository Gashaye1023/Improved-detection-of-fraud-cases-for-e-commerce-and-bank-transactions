import shap
import matplotlib.pyplot as plt
import joblib
import os
import pandas as pd
import numpy as np
import logging
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from typing import Union, Tuple
import yaml
import sys

# Configure logging with file handler
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/shap_analysis.log')
    ]
)
logger = logging.getLogger(__name__)

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        logger.info(f"Configuration loaded from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        sys.exit(1)

def load_model_and_data(model_path: str, data_path: str) -> Tuple[Union[LogisticRegression, RandomForestClassifier], pd.DataFrame]:
    """Load the best model and preprocessed data."""
    try:
        if not os.path.exists(model_path):
            logger.error(f"Model file not found at {model_path}")
            sys.exit(1)
        if not os.path.exists(data_path):
            logger.error(f"Data file not found at {data_path}")
            sys.exit(1)
        
        model = joblib.load(model_path)
        logger.info(f"Model loaded successfully from {model_path}")
        
        data = pd.read_csv(data_path)
        logger.info(f"Data loaded successfully from {data_path}")
        return model, data
    except Exception as e:
        logger.error(f"Failed to load model or data: {e}")
        sys.exit(1)

def prepare_data(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Prepare features and target for SHAP analysis."""
    try:
        if 'class' not in data.columns:
            logger.error("Target column 'class' not found in data.")
            sys.exit(1)
        X = data.drop(columns=['class'])
        y = data['class']
        logger.info("Data prepared for SHAP analysis")
        return X, y
    except Exception as e:
        logger.error(f"Error preparing data: {e}")
        sys.exit(1)

def generate_shap_plots(model: Union[LogisticRegression, RandomForestClassifier], X: pd.DataFrame, output_dir: str) -> Union[np.ndarray, list]:
    """Generate and save SHAP summary, beeswarm, dependence, and force plots."""
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        # Select appropriate SHAP explainer
        if isinstance(model, LogisticRegression):
            explainer = shap.LinearExplainer(model, X)
            shap_values = explainer.shap_values(X)
            model_name = "Logistic Regression"
        elif isinstance(model, RandomForestClassifier):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)
            model_name = "Random Forest"
        else:
            logger.error(f"Unsupported model type: {type(model)}. Expected LogisticRegression or RandomForestClassifier.")
            sys.exit(1)
        
        # Summary Plot (Bar)
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values[1] if isinstance(model, RandomForestClassifier) else shap_values, 
                         X, feature_names=X.columns, plot_type="bar", show=False)
        plt.title(f"SHAP Summary Plot (Bar) for {model_name} Model")
        summary_bar_path = os.path.join(output_dir, 'shap_summary_bar.png')
        plt.savefig(summary_bar_path, bbox_inches='tight')
        plt.close()
        logger.info(f"SHAP summary bar plot saved to {summary_bar_path}")
        
        # Beeswarm Plot
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values[1] if isinstance(model, RandomForestClassifier) else shap_values, 
                         X, feature_names=X.columns, show=False)
        plt.title(f"SHAP Beeswarm Plot for {model_name} Model")
        beeswarm_path = os.path.join(output_dir, 'shap_beeswarm_plot.png')
        plt.savefig(beeswarm_path, bbox_inches='tight')
        plt.close()
        logger.info(f"SHAP beeswarm plot saved to {beeswarm_path}")
        
        # Dependence Plot for top feature
        top_feature = X.columns[np.abs(shap_values[1] if isinstance(model, RandomForestClassifier) else shap_values).mean(axis=0).argmax()]
        plt.figure(figsize=(10, 6))
        shap.dependence_plot(top_feature, shap_values[1] if isinstance(model, RandomForestClassifier) else shap_values, 
                             X, show=False)
        plt.title(f"SHAP Dependence Plot for {top_feature}")
        dependence_path = os.path.join(output_dir, f'shap_dependence_{top_feature}.png')
        plt.savefig(dependence_path, bbox_inches='tight')
        plt.close()
        logger.info(f"SHAP dependence plot saved to {dependence_path}")
        
        # Force Plot for a single prediction
        instance_to_explain = X.iloc[0]
        shap_value_instance = explainer.shap_values(instance_to_explain)
        plt.figure()
        shap.force_plot(explainer.expected_value[1] if isinstance(model, RandomForestClassifier) else explainer.expected_value, 
                        shap_value_instance[1] if isinstance(model, RandomForestClassifier) else shap_value_instance, 
                        instance_to_explain, matplotlib=True, show=False)
        plt.title("SHAP Force Plot for a Single Prediction")
        force_plot_path = os.path.join(output_dir, 'shap_force_plot.png')
        plt.savefig(force_plot_path, bbox_inches='tight')
        plt.close()
        logger.info(f"SHAP force plot saved to {force_plot_path}")
        
        return shap_values
    except Exception as e:
        logger.error(f"Error generating SHAP plots: {e}")
        sys.exit(1)

def interpret_shap_results(shap_values: Union[np.ndarray, list], X: pd.DataFrame, output_dir: str) -> pd.DataFrame:
    """Interpret SHAP results and save report."""
    try:
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'mean_abs_shap': np.abs(shap_values[1] if isinstance(shap_values, list) else shap_values).mean(axis=0)
        })
        feature_importance = feature_importance.sort_values(by='mean_abs_shap', ascending=False)
        
        interpretation = (
            "# SHAP Analysis Interpretation\n\n"
            "## Key Drivers of Fraud\n"
            "The SHAP summary and beeswarm plots highlight the most influential features driving fraud predictions:\n"
        )
        for i, row in feature_importance.head(5).iterrows():
            interpretation += f"- **{row['feature']}**: Mean absolute SHAP value of {row['mean_abs_shap']:.4f}. "
            interpretation += "High values indicate strong influence on fraud prediction.\n"
        
        interpretation += (
            "\n## Local Interpretation\n"
            "The SHAP force plot shows how features contribute to individual predictions. "
            "Positive SHAP values increase fraud likelihood, negative values decrease it.\n"
            "\n## Dependence Analysis\n"
            f"The dependence plot for the top feature ({feature_importance.iloc[0]['feature']}) shows its interaction with other features, "
            "revealing patterns in fraud risk.\n"
        )
        
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'shap_interpretation.md')
        with open(output_path, 'w') as f:
            f.write(interpretation)
        logger.info(f"SHAP interpretation saved to {output_path}")
        
        return feature_importance
    except Exception as e:
        logger.error(f"Error interpreting SHAP results: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Load configuration
    config = load_config('config.yaml')
    
    # Configuration
    model_path = config['model_dir'] + '/best_fraud_model.pkl'
    data_path = config['processed_data_path']
    plot_dir = config['plot_dir']
    report_dir = config['report_dir']
    
    # Log current working directory and files
    logger.info(f"Current working directory: {os.getcwd()}")
    logger.info(f"Files in directory: {os.listdir()}")
    
    # Load model and data
    best_model, data = load_model_and_data(model_path, data_path)
    
    # Prepare data
    X, y = prepare_data(data)
    
    # Generate SHAP plots
    shap_values = generate_shap_plots(best_model, X, plot_dir)
    
    # Interpret results
    feature_importance = interpret_shap_results(shap_values, X, report_dir)
    
    logger.info("SHAP analysis completed successfully.")