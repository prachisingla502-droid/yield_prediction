import sys
sys.path.append('.')

import json
import torch
import streamlit as st
from src.model import YieldNet
from src.preprocess import load_data

st.title("Chemical Reaction Yield Prediction")
st.write("Predict chemical reaction yield (%) using Neural Network Regression.")

temp = st.slider("Temperature (°C)", min_value=100.0, max_value=500.0, value=392.0)
pressure = st.slider("Pressure (bar)", min_value=1.0, max_value=100.0, value=18.0)
conc = st.slider("Concentration (M)", min_value=0.1, max_value=5.0, value=1.74)

if st.button("Predict Yield"):
    try:
        # Load hyperparameter configuration
        with open("configs/config.json", "r") as f:
            config = json.load(f)

        # Initialize model with configuration
        model = YieldNet(config)
        
        # Load trained model weights
        checkpoint = torch.load("results/model.pth", map_location=torch.device('cpu'))
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
            
        model.eval()

        # Load scaler to scale input features correctly
        _, _, _, _, scaler = load_data(config)
        raw_inputs = [[temp, pressure, conc]]
        scaled_inputs = scaler.transform(raw_inputs)

        # Make prediction
        input_tensor = torch.tensor(scaled_inputs, dtype=torch.float32)
        with torch.no_grad():
            prediction = model(input_tensor).item()

        st.success(f"Predicted Yield: {prediction:.2f}%")
    except Exception as e:
        st.error(f"Error running prediction: {e}")
