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

        # Initialize model with config directly
        try:
            model = YieldNet(config)
        except Exception:
            model = YieldNet()

        # Load model weights
        checkpoint = torch.load("results/model.pth", map_location=torch.device('cpu'))
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        else:
            model.load_state_dict(checkpoint)
            
        model.eval()

        # Min-Max Normalization (matching training bounds: 100-500°C, 1-100 bar, 0.1-5.0 M)
        temp_norm = (temp - 100.0) / (500.0 - 100.0)
        pressure_norm = (pressure - 1.0) / (100.0 - 1.0)
        conc_norm = (conc - 0.1) / (5.0 - 0.1)

        input_tensor = torch.tensor([[temp_norm, pressure_norm, conc_norm]], dtype=torch.float32)

        with torch.no_grad():
            raw_pred = model(input_tensor).item()

        yield_pred = max(0.0, min(100.0, raw_pred))
        st.success(f"Predicted Yield: {yield_pred:.2f}%")

    except Exception as e:
        st.error(f"Error making prediction: {e}")
