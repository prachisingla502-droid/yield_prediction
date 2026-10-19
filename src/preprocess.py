"""Train/validation/test split and strict feature scaling.

"Strict" means the scaler is fitted on the training split ONLY and then applied
unchanged to validation and test data, so no information from held-out rows
leaks into training.
"""
from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


@dataclass
class Prepared:
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray       # standardised target (what the network trains on)
    y_val: np.ndarray
    y_test_pct: np.ndarray    # untouched target in % (what metrics are computed on)
    x_scaler: StandardScaler | None
    y_scaler: StandardScaler


def split(df, fractions, seed):
    """Return (train, val, test) DataFrames."""
    holdout = fractions["val"] + fractions["test"]
    train, rest = train_test_split(df, test_size=holdout, random_state=seed)
    val, test = train_test_split(rest, test_size=fractions["test"] / holdout, random_state=seed)
    return train, val, test


def prepare(train, val, test, features, target, scale_features=True):
    """Scale features (optional, for the ablation) and the target, fitting on train only."""
    X_tr, X_va, X_te = (d[features].to_numpy(np.float32) for d in (train, val, test))
    x_scaler = None
    if scale_features:
        x_scaler = StandardScaler().fit(X_tr)
        X_tr, X_va, X_te = (x_scaler.transform(X).astype(np.float32) for X in (X_tr, X_va, X_te))

    y_scaler = StandardScaler().fit(train[[target]].to_numpy())
    to_z = lambda d: y_scaler.transform(d[[target]].to_numpy()).ravel().astype(np.float32)
    return Prepared(X_tr, X_va, X_te, to_z(train), to_z(val),
                    test[target].to_numpy(), x_scaler, y_scaler)
