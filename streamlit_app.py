import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import yaml
import os
import logging
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/streamlit_app.log')
    ]
)
logger = logging.getLogger(__name__)

# Load configuration
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Load model and data
@st.cache_resource
def load_model_and_data(model_path, data_path):
    try:
        model = joblib.load(model_path)
        data = pd.read_csv(data_path)
        logger.info("Model and data loaded successfully for Streamlit app")
        return model, data
    except Exception as e:
        logger.error(f"Error loading model or data: {e}")
        st.error(f"Failed to load model or data: {e}")
        return None, None

# Generate SHAP plots
@st.cache_data
def generate_shap_plot(_model, X, plot_type, feature=None):
    try:
        if isinstance(_model, LogisticRegression):
            explainer = shap.LinearExplainer(_model, X)
            shap_values = explainer.shap_values(X)
            expected_value = explainer.expected_value
        else:
            explainer = shap.TreeExplainer(_model)
            shap_values = explainer.shap_values(X)
            expected_value = explainer.expected_value[1]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        if plot_type == "summary_bar":
            shap.summary_plot(shap_values[1] if isinstance(_model, RandomForestClassifier) else shap_values, 
                             X, plot_type="bar", show=False)
            plt.title(f"SHAP Summary Plot (Bar)")
        elif plot_type == "beeswarm":
            shap.summary_plot(shap_values[1] if isinstance(_model, RandomForestClassifier) else shap_values, 
                             X, show=False)
            plt.title(f"SHAP Beeswarm Plot")
        elif plot_type == "dependence" and feature:
            shap.dependence_plot(feature, shap_values[1] if isinstance(_model, RandomForestClassifier) else shap_values, 
                                 X, show=False)
            plt.title(f"SHAP Dependence Plot for {feature}")
        elif plot_type == "force":
            instance = X.iloc[0]
            shap_values_instance = explainer.shap_values(instance)
            shap.force_plot(expected_value, 
                            shap_values_instance[1] if isinstance(_model, RandomForestClassifier) else shap_values_instance, 
                            instance, matplotlib=True, show=False)
            plt.title("SHAP Force Plot for a Single Prediction")
        
        return fig
    except Exception as e:
        logger.error(f"Error generating SHAP plot: {e}")
        st.error(f"Error generating SHAP plot: {e}")
        return None

# Streamlit app
st.title("Fraud Detection Dashboard")
st.write("Explore fraud predictions and feature impacts for e-commerce transactions.")

# Load model and data
model, data = load_model_and_data(config['model_dir'] + '/best_fraud_model.pkl', config['processed_data_path'])

if model is not None and data is not None:
    X = data.drop(columns=['class'])
    y = data['class']
    
    # Display model type
    model_name = "Logistic Regression" if isinstance(model, LogisticRegression) else "Random Forest"
    st.write(f"**Model Used**: {model_name}")
    
    # Predict and show sample predictions
    predictions = model.predict(X)
    st.write("### Sample Predictions")
    sample_df = pd.DataFrame({
        'Prediction': ['Fraud' if pred == 1 else 'Not Fraud' for pred in predictions[:10]],
        'True Label': ['Fraud' if label == 1 else 'Not Fraud' for label in y[:10]]
    })
    st.dataframe(sample_df)
    
    # SHAP Plots
    st.write("### SHAP Explanations")
    plot_type = st.selectbox("Select SHAP Plot Type", ["Summary (Bar)", "Beeswarm", "Dependence", "Force"])
    
    if plot_type == "Dependence":
        feature = st.selectbox("Select Feature for Dependence Plot", X.columns)
        fig = generate_shap_plot(model, X, "dependence", feature)
    else:
        fig = generate_shap_plot(model, X, plot_type.lower().replace(" ", "_"))
    
    if fig:
        st.pyplot(fig)
    
    # Feature Distribution
    st.write("### Feature Distribution")
    feature = st.selectbox("Select Feature to Visualize", X.columns)
    fig, ax = plt.subplots()
    X[feature].hist(bins=30, ax=ax)
    ax.set_title(f"Distribution of {feature}")
    st.pyplot(fig)
    
    # Business Impact
    st.write("### Business Impact")
    st.markdown("""
    - **Reduced Financial Losses**: Accurate fraud detection minimizes chargebacks and losses.
    - **Improved Customer Trust**: Fewer false positives reduce customer friction.
    - **Regulatory Compliance**: SHAP explanations ensure transparency for audits.
    """)

else:
    st.error("Failed to load model or data. Please check logs.")