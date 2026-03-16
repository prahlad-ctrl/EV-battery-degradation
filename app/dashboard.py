import streamlit as st
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from lstm_model import BatteryLSTM

st.set_page_config(page_title="EV Battery Health Predictor", layout="centered")
st.title("EV Battery Degradation & Health Predictor")
st.write("Upload the previous 10 cycles of battery sensor logs to predict current State of Health (SOH).")

@st.cache_resource
def load_model():
    model = BatteryLSTM(input_size=5)
    model_path = os.path.join(os.path.dirname(__file__), '..','model', 'battery_lstm.pth')
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

uploaded_file = st.file_uploader("Upload Battery Data (CSV format)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Uploaded Data Preview")
    st.dataframe(df.tail(3))
    
    required_cols = ['cycle', 'Re', 'Rct', 'Capacity_Fade', 'ambient_temperature']
    if not all(col in df.columns for col in required_cols):
        st.error(f"Error: CSV must contain the following columns: {required_cols}")
    elif len(df) < 10:
        st.error("Error: The model requires at least 10 cycles of historical data to make a prediction.")
    else:
        if st.button("Predict Battery Health"):
            with st.spinner('Analyzing sensor sequences...'):
                scaler = MinMaxScaler()
                scaled_data = scaler.fit_transform(df[required_cols])

                recent_sequence = scaled_data[-10:] 

                tensor_input = torch.tensor(recent_sequence, dtype=torch.float32).unsqueeze(0)

                with torch.no_grad():
                    prediction = model(tensor_input).item()
                
                health_percentage = prediction * 100
                
                st.write("---")
                st.subheader("Prediction Results")
                
                if health_percentage >= 85:
                    risk = "Low"
                    color = "green"
                elif health_percentage >= 75:
                    risk = "Medium"
                    color = "orange"
                else:
                    risk = "High (Replacement Recommended)"
                    color = "red"
                
                col1, col2 = st.columns(2)
                col1.metric("State of Health (SOH)", f"{health_percentage:.1f}%")
                col2.markdown(f"**Risk Level:** <span style='color:{color}'>{risk}</span>", unsafe_allow_html=True)