import sys
sys.path.append('.')

import json
import torch
import streamlit as st
from src.model import YieldNet

st.set_page_config(page_title="Yield Prediction", layout="centered")

st.title("Chemical Reaction Yield Prediction")
st.markdown("Predict chemical reaction yield (%) using Neural Network Regression.")

# User inputs
temp = st.slider("Temperature (°C)", min_value=100.0, max_value=500.0, value=392.0)
pressure = st.slider("Pressure (bar)", min_value=1.0, max_value=100.0, value=18.0)
conc = st.slider("Concentration (M)", min_value=0.1, max_value=5.0, value=1.74)

if st.button("Predict Yield", type="primary"):
    try:
        # Load hyperparameter configuration
        with open("configs/config.json", "r") as f:
            config = json.load(f)

        # Initialize model architecture
        model = YieldNet(config)
        
        # Load weights on CPU
        checkpoint = torch.load("results/model.pth", map_location=torch.device('cpu'))
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
            
        model.eval()

        # Min-Max Normalization matching feature space (100–500, 1–100, 0.1–5.0)
        temp_norm = (temp - 100.0) / (500.0 - 100.0)
        pressure_norm = (pressure - 1.0) / (100.0 - 1.0)
        conc_norm = (conc - 0.1) / (5.0 - 0.1)

        input_tensor = torch.tensor([[temp_norm, pressure_norm, conc_norm]], dtype=torch.float32)

        with torch.no_grad():
            raw_pred = model(input_tensor).item()

        # Clamp yield prediction between 0% and 100%
        yield_pred = max(0.0, min(100.0, raw_pred))

        st.metric(label="Predicted Yield", value=f"{yield_pred:.2f}%")

    except Exception as e:
        st.error(f"Error making prediction: {e}")
