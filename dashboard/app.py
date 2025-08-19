## dashboard/app.py
import streamlit as st
import pandas as pd
from src.fraud_predictor import FraudPredictor
from src.visualizer import Visualizer

st.title("Fraud Detection Dashboard")

predictor = FraudPredictor()
visualizer = Visualizer()

uploaded_file = st.file_uploader("Upload Fraud Data (CSV)", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Data Preview", df.head())
    
    if st.button("Analyze"):
        probs = predictor.predict(df)
        df['fraud_probability'] = probs
        st.write("Predictions", df[['user_id', 'fraud_probability']])
        
        fig = plt.figure(figsize=(10, 6))
        sns.histplot(df['fraud_probability'])
        st.pyplot(fig)
        
        X, _, _ = predictor.process_and_train({'fraud': 'data/Fraud_Data.csv', 'ip': 'data/IpAddress_to_Country.csv'})
        shap_values = predictor.trainer.get_shap_values(X)
        visualizer.plot_shap_summary(shap_values, X, 'shap.png')
        st.image('shap.png')
