import json
from pathlib import Path

import streamlit as st
import torch

from src.model import YieldNet


# --------------------------------------------------
# Configuration
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "configs" / "config.json"
MODEL_PATH = BASE_DIR / "results" / "model.pth"

FEATURE_RANGES = {
    "temperature": (100.0, 500.0),
    "pressure": (1.0, 100.0),
    "concentration": (0.1, 5.0),
}

st.set_page_config(
    page_title="Chemical Yield Predictor",
    page_icon="🧪",
    layout="centered",
    initial_sidebar_state="expanded",
)

DEVICE = torch.device("cpu")


# --------------------------------------------------
# Model loading
# --------------------------------------------------
@st.cache_resource
def load_model():
    """Load the trained model once and reuse it across reruns."""

    if not CONFIG_PATH.is_file():
        raise FileNotFoundError(f"Configuration not found: {CONFIG_PATH}")

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {MODEL_PATH}")

    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        config = json.load(file)

    model = YieldNet(config)

    # Load trusted, locally stored model weights.
    try:
        checkpoint = torch.load(
            MODEL_PATH,
            map_location=DEVICE,
            weights_only=True,
        )
    except TypeError:
        # Compatibility with older PyTorch versions.
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    # Support checkpoints saved from DataParallel models.
    if (
        isinstance(state_dict, dict)
        and state_dict
        and all(key.startswith("module.") for key in state_dict)
    ):
        state_dict = {
            key.removeprefix("module."): value
            for key, value in state_dict.items()
        }

    model.load_state_dict(state_dict, strict=True)
    model.to(DEVICE)
    model.eval()

    return model


# --------------------------------------------------
# Preprocessing and prediction
# --------------------------------------------------
def normalize_feature(value: float, feature: str) -> float:
    """Apply the same Min-Max scaling used during training."""

    minimum, maximum = FEATURE_RANGES[feature]
    return (value - minimum) / (maximum - minimum)


def predict_yield(
    model: torch.nn.Module,
    temperature: float,
    pressure: float,
    concentration: float,
) -> float:
    """Predict reaction yield from the three input features."""

    features = [
        normalize_feature(temperature, "temperature"),
        normalize_feature(pressure, "pressure"),
        normalize_feature(concentration, "concentration"),
    ]

    inputs = torch.tensor([features], dtype=torch.float32, device=DEVICE)

    with torch.inference_mode():
        output = model(inputs)

    if output.numel() != 1:
        raise ValueError(
            f"Expected one prediction, received {output.numel()} values."
        )

    raw_prediction = output.item()

    if not torch.isfinite(torch.tensor(raw_prediction)).item():
        raise ValueError("The model returned a non-finite prediction.")

    # Preserve the original application's 0–100% display bounds.
    # Note that clipping does not guarantee the model is well calibrated.
    return max(0.0, min(100.0, raw_prediction))


# --------------------------------------------------
# User interface
# --------------------------------------------------
st.title("🧪 Chemical Reaction Yield Prediction")

st.markdown(
    """
    Estimate chemical reaction yield using a trained neural network
    regression model.

    Adjust the reaction conditions below and click **Predict Yield**.
    """
)

with st.sidebar:
    st.header("About")
    st.write(
        "This application estimates reaction yield from temperature, "
        "pressure, and reactant concentration."
    )

    st.caption("Model: YieldNet")
    st.caption("Inference device: CPU")

st.subheader("Reaction Conditions")

col1, col2 = st.columns(2)

with col1:
    temperature = st.slider(
        "Temperature (°C)",
        min_value=100.0,
        max_value=500.0,
        value=392.0,
        step=1.0,
    )

with col2:
    pressure = st.slider(
        "Pressure (bar)",
        min_value=1.0,
        max_value=100.0,
        value=18.0,
        step=1.0,
    )

concentration = st.slider(
    "Concentration (M)",
    min_value=0.1,
    max_value=5.0,
    value=1.74,
    step=0.01,
    help="Reactant concentration in mol/L.",
)

st.divider()

predict_clicked = st.button(
    "Predict Yield",
    type="primary",
    use_container_width=True,
)

if predict_clicked:
    try:
        with st.spinner("Analyzing reaction conditions..."):
            model = load_model()

            prediction = predict_yield(
                model=model,
                temperature=temperature,
                pressure=pressure,
                concentration=concentration,
            )

        st.subheader("Prediction Result")

        st.metric(
            label="Predicted Reaction Yield",
            value=f"{prediction:.2f}%",
        )

        st.progress(prediction / 100.0)

        if prediction < 30:
            st.info("The model predicts a relatively low yield.")
        elif prediction < 70:
            st.info("The model predicts a moderate yield.")
        else:
            st.success("The model predicts a relatively high yield.")

        with st.expander("View input details"):
            st.write(
                {
                    "Temperature (°C)": temperature,
                    "Pressure (bar)": pressure,
                    "Concentration (M)": concentration,
                }
            )

        st.caption(
            "This is a model estimate, not an experimental measurement. "
            "Prediction accuracy depends on the training data and "
            "model validation."
        )

    except FileNotFoundError as error:
        st.error(str(error))
        st.info(
            "Check that configs/config.json and results/model.pth "
            "exist in your project."
        )

    except (json.JSONDecodeError, RuntimeError, ValueError, TypeError) as error:
        st.error(f"Unable to make a prediction: {error}")

    except Exception:
        st.error(
            "An unexpected error occurred. Check the application logs "
            "for details."
        )
