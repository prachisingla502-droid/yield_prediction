"""Predict yield for new reaction conditions with the trained model.

    python predict.py --temperature 390 --pressure 25 --concentration 1.0
"""
import argparse
from pathlib import Path

import joblib
import numpy as np
import torch

from src.model import YieldNet, predict

MODEL_DIR = Path(__file__).parent / "results" / "model"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--temperature", type=float, required=True, help="K (trained range 300-450)")
    ap.add_argument("--pressure", type=float, required=True, help="bar (trained range 1-50)")
    ap.add_argument("--concentration", type=float, required=True, help="mol/L (trained range 0.1-2.0)")
    args = ap.parse_args()

    bundle = joblib.load(MODEL_DIR / "scalers.joblib")
    model = YieldNet(len(bundle["features"]), bundle["hidden_layers"])
    model.load_state_dict(torch.load(MODEL_DIR / "yieldnet.pt"))

    X = np.array([[args.temperature, args.pressure, args.concentration]], dtype=np.float32)
    z = predict(model, bundle["x_scaler"].transform(X).astype(np.float32))
    y = float(bundle["y_scaler"].inverse_transform(z.reshape(-1, 1))[0, 0])
    y = min(max(y, 0.0), 100.0)   # linear output is unbounded; yield is physically 0-100 %
    print(f"Predicted yield: {y:.2f} %")


if __name__ == "__main__":
    main()
