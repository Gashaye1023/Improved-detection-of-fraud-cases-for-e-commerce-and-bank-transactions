import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.fraud_predictor import FraudPredictor
from src.visualizer import Visualizer

def load_data(uploaded_file):
    return pd.read_csv(uploaded_file)

def display_predictions(df, probs):
    df['fraud_probability'] = probs
    st.write("Predictions", df[['user_id', 'fraud_probability']])
    
    fig = plt.figure(figsize=(10, 6))
    sns.histplot(df['fraud_probability'])
    st.pyplot(fig)

def main():
    st.title("Fraud Detection Dashboard")
    st.write("Welcome to the Fraud Detection Dashboard for e-commerce and bank transactions.")
    
    predictor = FraudPredictor()
    visualizer = Visualizer()

    uploaded_file = st.file_uploader("Upload Fraud Data (CSV)", type="csv")

    if uploaded_file:
        try:
            df = load_data(uploaded_file)
            st.write("Data Preview", df.head())
            
            if st.button("Analyze"):
                with st.spinner("Analyzing data..."):
                    probs = predictor.predict(df)
                    display_predictions(df, probs)
                    
                    X, _, _ = predictor.process_and_train({'fraud': 'data/Fraud_Data.csv', 'ip': 'data/IpAddress_to_Country.csv'})
                    shap_values = predictor.trainer.get_shap_values(X)
                    visualizer.plot_shap_summary(shap_values, X, 'shap.png')
                    st.image('shap.png')
        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()