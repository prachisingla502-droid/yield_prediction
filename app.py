import sys
sys.path.append('.')

import streamlit as st
import torch
from src.model import YieldNet

st.title("Chemical Reaction Yield Prediction")
st.write("Predict chemical reaction yield (%) using Neural Network Regression.")

temp = st.slider("Temperature (°C)", min_value=100.0, max_value=500.0, value=392.0)
pressure = st.slider("Pressure (bar)", min_value=1.0, max_value=100.0, value=25.0)
conc = st.slider("Concentration (M)", min_value=0.1, max_value=5.0, value=1.0)

if st.button("Predict Yield"):
    # Load model and run prediction
    model = YieldNet()
    try:
        model.load_state_dict(torch.load("results/model.pth"))
        model.eval()
        
        # Scale inputs (example scaling step)
        inputs = torch.tensor([[temp, pressure, conc]], dtype=torch.float32)
        with torch.no_grad():
            prediction = model(inputs).item()
        
        st.success(f"Predicted Yield: {prediction:.2f}%")
    except Exception as e:
        st.error(f"Error running prediction: {e}")
