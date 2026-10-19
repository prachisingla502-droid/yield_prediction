"""Simulated reaction-yield dataset.

Reaxys is a licensed database, so this project uses a physically motivated
simulator instead:

    yield (%) = 100 * conversion(T, P, C) * selectivity(T, C) + measurement noise

* conversion follows Arrhenius kinetics (rate rises exponentially with T),
  a saturating pressure effect, and a square-root concentration order;
* selectivity falls sharply above ~420 K (side reaction / decomposition) and
  mildly at high concentration (more by-products).

The result has an optimum temperature window, which a straight line or a
quadratic cannot follow well -- the reason a neural network is used.
"""
import numpy as np
import pandas as pd

R = 8.314           # gas constant, J/(mol K)
EA = 60_000.0       # activation energy, J/mol
A_T = 3.4e8         # pre-exponential factor x fixed reaction time (dimensionless)
K_P = 5.0           # bar; pressure at which the pressure effect is half-saturated
T_DECOMP = 420.0    # K; onset of the side reaction
T_WIDTH = 12.0      # K; sharpness of that onset
C_SIDE = 0.1        # strength of the concentration-driven side reaction

FEATURES = ["temperature_K", "pressure_bar", "concentration_molL"]
TARGET = "yield_pct"
RANGES = {
    "temperature_K": (300.0, 450.0),
    "pressure_bar": (1.0, 50.0),
    "concentration_molL": (0.1, 2.0),
}


def true_yield(T, P, C):
    """Noise-free yield (%) for temperature T [K], pressure P [bar], concentration C [mol/L]."""
    k = A_T * np.exp(-EA / (R * T))
    conversion = 1.0 - np.exp(-k * (P / (P + K_P)) * np.sqrt(C))
    selectivity = 1.0 / (1.0 + np.exp((T - T_DECOMP) / T_WIDTH))
    selectivity = selectivity / (1.0 + C_SIDE * C ** 2)
    return 100.0 * conversion * selectivity


def generate(n_samples, noise_std, seed):
    rng = np.random.default_rng(seed)
    T = rng.uniform(*RANGES["temperature_K"], n_samples)
    P = rng.uniform(*RANGES["pressure_bar"], n_samples)
    C = rng.uniform(*RANGES["concentration_molL"], n_samples)
    y = true_yield(T, P, C) + rng.normal(0.0, noise_std, n_samples)
    return pd.DataFrame({
        "temperature_K": T.round(1),
        "pressure_bar": P.round(2),
        "concentration_molL": C.round(3),
        TARGET: np.clip(y, 0.0, 100.0).round(2),
    })
