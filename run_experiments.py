"""Run the whole project end to end.

    python run_experiments.py

Produces data/reaction_yield.csv, results/metrics.json, results/*.csv,
results/figures/*.png and the trained model in results/model/.
"""
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from src import plots
from src.data_generator import FEATURES, TARGET, generate, true_yield
from src.metrics import regression_metrics
from src.model import count_parameters, predict
from src.preprocess import prepare, split
from src.train import train_model

ROOT = Path(__file__).parent
FIG = ROOT / "results" / "figures"


def nn_predict_pct(model, prep, X_raw):
    X = prep.x_scaler.transform(X_raw).astype(np.float32) if prep.x_scaler else X_raw.astype(np.float32)
    return prep.y_scaler.inverse_transform(predict(model, X).reshape(-1, 1)).ravel()


def run_nn(train, val, test, features, cfg, seed, hidden=None, scale_features=True):
    prep = prepare(train, val, test, features, TARGET, scale_features)
    model_cfg = dict(cfg["model"], **({"hidden_layers": hidden} if hidden else {}))
    t0 = time.perf_counter()
    model, hist = train_model(prep.X_train, prep.y_train, prep.X_val, prep.y_val, model_cfg, cfg["training"], seed)
    seconds = time.perf_counter() - t0
    y_pred = nn_predict_pct(model, prep, test[features].to_numpy(np.float32))
    m = regression_metrics(prep.y_test_pct, y_pred)
    m.update(epochs_run=len(hist["val_loss"]), best_epoch=hist["best_epoch"] + 1,
             parameters=count_parameters(model), train_seconds=round(seconds, 2))
    return model, prep, hist, y_pred, m


def baseline_models(cfg, seed):
    return {
        "Linear Regression": make_pipeline(StandardScaler(), LinearRegression()),
        "Polynomial Regression (deg 2, RSM-style)": make_pipeline(
            StandardScaler(), PolynomialFeatures(cfg["baselines"]["polynomial_degree"]), LinearRegression()),
        "Random Forest": RandomForestRegressor(n_estimators=cfg["baselines"]["random_forest_trees"],
                                               random_state=seed, n_jobs=-1),
    }


def main():
    cfg = json.loads((ROOT / "configs" / "config.json").read_text())
    seed = cfg["seed"]
    np.random.seed(seed)
    torch.manual_seed(seed)
    FIG.mkdir(parents=True, exist_ok=True)
    (ROOT / "results" / "model").mkdir(parents=True, exist_ok=True)
    (ROOT / "data").mkdir(exist_ok=True)
    out = {"config": cfg}

    # ---- 1. Dataset -------------------------------------------------------
    df = generate(cfg["data"]["n_samples"], cfg["data"]["noise_std"], seed)
    df.to_csv(ROOT / cfg["data"]["path"], index=False)
    df.describe().T.round(3).to_csv(ROOT / "results" / "data_summary.csv")
    out["dataset"] = {
        "n_samples": len(df),
        "missing_values": int(df.isna().sum().sum()),
        "summary": df.describe().T.round(3).to_dict(orient="index"),
        "correlation_with_yield": df.corr()[TARGET].drop(TARGET).round(3).to_dict(),
        "share_yield_below_10pct": round(float((df[TARGET] < 10).mean()), 3),
    }
    print(f"[data] {len(df)} rows, yield mean {df[TARGET].mean():.1f}%")

    plots.flowchart(FIG / "fig01_pipeline_flowchart.png")
    plots.distributions(df, FEATURES, TARGET, FIG / "fig02_distributions.png")
    plots.correlation(df, FEATURES + [TARGET], FIG / "fig03_correlation.png")
    plots.yield_vs_features(df, FEATURES, TARGET, FIG / "fig04_yield_vs_features.png")

    # ---- 2. Split + strict scaling ----------------------------------------
    train, val, test = split(df, cfg["split"], seed)
    dev = pd.concat([train, val])
    out["split_sizes"] = {"train": len(train), "val": len(val), "test": len(test)}
    prep_main = prepare(train, val, test, FEATURES, TARGET)
    out["scaler"] = {"mean": dict(zip(FEATURES, prep_main.x_scaler.mean_.round(4).tolist())),
                     "std": dict(zip(FEATURES, prep_main.x_scaler.scale_.round(4).tolist())),
                     "target_mean": round(float(prep_main.y_scaler.mean_[0]), 4),
                     "target_std": round(float(prep_main.y_scaler.scale_[0]), 4)}
    plots.scaling_effect(train[FEATURES].to_numpy(), prep_main.X_train, FEATURES, FIG / "fig05_scaling_effect.png")

    # ---- 3. Main model ----------------------------------------------------
    model, prep, hist, y_pred, m_nn = run_nn(train, val, test, FEATURES, cfg, seed)
    out["neural_network"] = m_nn
    print(f"[nn] test R2 {m_nn['r2']:.4f}  MAE {m_nn['mae']:.3f}  RMSE {m_nn['rmse']:.3f}  epochs {m_nn['epochs_run']}")
    torch.save(model.state_dict(), ROOT / "results" / "model" / "yieldnet.pt")
    joblib.dump({"x_scaler": prep.x_scaler, "y_scaler": prep.y_scaler, "features": FEATURES,
                 "hidden_layers": cfg["model"]["hidden_layers"]}, ROOT / "results" / "model" / "scalers.joblib")

    y_true = prep.y_test_pct
    resid = y_true - y_pred
    sd = float(resid.std())
    bands = pd.cut(y_pred, [-np.inf, 20, 60, np.inf], labels=["< 20 %", "20-60 %", "> 60 %"])
    out["residuals"] = {
        "mean": round(float(resid.mean()), 4),
        "std": round(sd, 4),
        "share_within_2std": round(float((np.abs(resid) <= 2 * sd).mean()), 4),
        "corr_abs_resid_vs_pred": round(float(np.corrcoef(np.abs(resid), y_pred)[0, 1]), 4),
        "max_abs": round(float(np.abs(resid).max()), 3),
        "pred_min": round(float(y_pred.min()), 3),
        "pred_max": round(float(y_pred.max()), 3),
        "by_predicted_band": {str(b): {"n": int((bands == b).sum()),
                                       "mean": round(float(resid[bands == b].mean()), 3),
                                       "std": round(float(resid[bands == b].std()), 3)}
                              for b in bands.categories},
        "noise_floor_std": cfg["data"]["noise_std"],
    }
    pd.DataFrame({**{f: test[f].to_numpy() for f in FEATURES}, "actual": y_true,
                  "predicted": y_pred.round(3), "residual": resid.round(3)}).to_csv(
        ROOT / "results" / "test_predictions.csv", index=False)
    plots.loss_curves(hist, FIG / "fig06_loss_curves.png")
    plots.pred_vs_actual(y_true, y_pred, m_nn["r2"], FIG / "fig07_pred_vs_actual.png")
    plots.loss_and_parity(hist, y_true, y_pred, m_nn["r2"], FIG / "fig13_training_and_fit.png")
    plots.residuals(y_pred, resid, cfg["data"]["noise_std"], FIG / "fig08_residuals.png")

    # noise-free ceiling: how good is the *true* function on the noisy test labels?
    ceiling = regression_metrics(y_true, true_yield(*(test[f].to_numpy() for f in FEATURES)))
    out["noise_ceiling"] = ceiling

    # ---- 4. Comparison methods --------------------------------------------
    comparison = {"Neural Network (YieldNet)": m_nn}
    fitted = {}
    for name, est in baseline_models(cfg, seed).items():
        t0 = time.perf_counter()
        est.fit(train[FEATURES], train[TARGET])
        m = regression_metrics(y_true, est.predict(test[FEATURES]))
        m["train_seconds"] = round(time.perf_counter() - t0, 2)
        comparison[name] = m
        fitted[name] = est
        print(f"[baseline] {name}: R2 {m['r2']:.4f}  MAE {m['mae']:.3f}")
    out["comparison"] = comparison
    plots.model_comparison([(n, v["r2"], v["mae"]) for n, v in comparison.items()],
                           FIG / "fig09_model_comparison.png")
    poly_pred = fitted["Polynomial Regression (deg 2, RSM-style)"].predict(test[FEATURES])
    plots.residual_contrast([
        ("Polynomial deg 2: curved pattern = underfitting", poly_pred, y_true - poly_pred, plots.ORANGE),
        ("Neural network: random band around zero", y_pred, resid, plots.BLUE),
    ], FIG / "fig12_residual_contrast.png")

    # qualitative: one-factor-at-a-time response curves vs the true function
    T_grid = np.linspace(300, 450, 200)
    P_grid = np.linspace(1, 50, 200)
    def curve(T, P, C):
        X = np.column_stack([T, P, C]).astype(np.float32)
        Xdf = pd.DataFrame(X, columns=FEATURES)
        return [("True (simulator)", true_yield(T, P, C), plots.INK, "--"),
                ("Neural network", nn_predict_pct(model, prep, X), plots.BLUE, "-"),
                ("Polynomial deg 2", fitted["Polynomial Regression (deg 2, RSM-style)"].predict(Xdf), plots.ORANGE, "-"),
                ("Random forest", fitted["Random Forest"].predict(Xdf), plots.AQUA, "-")]
    plots.response_curves({
        "Yield vs temperature (P = 25 bar, C = 1.0 mol/L)":
            {"x": T_grid, "xlabel": "Temperature (K)", "series": curve(T_grid, np.full(200, 25.0), np.full(200, 1.0))},
        "Yield vs pressure (T = 395 K, C = 1.0 mol/L)":
            {"x": P_grid, "xlabel": "Pressure (bar)", "series": curve(np.full(200, 395.0), P_grid, np.full(200, 1.0))},
    }, FIG / "fig10_response_curves.png")
    # predicted optimum temperature at P=25, C=1
    X_line = np.column_stack([T_grid, np.full(200, 25.0), np.full(200, 1.0)]).astype(np.float32)
    nn_curve = nn_predict_pct(model, prep, X_line)
    poly_curve = fitted["Polynomial Regression (deg 2, RSM-style)"].predict(pd.DataFrame(X_line, columns=FEATURES))
    true_curve = true_yield(T_grid, 25.0, 1.0)
    out["optimum_temperature"] = {
        "true_T": round(float(T_grid[true_curve.argmax()]), 1), "true_yield": round(float(true_curve.max()), 2),
        "nn_T": round(float(T_grid[nn_curve.argmax()]), 1), "nn_yield": round(float(nn_curve.max()), 2),
        "poly_T": round(float(T_grid[poly_curve.argmax()]), 1), "poly_yield": round(float(poly_curve.max()), 2),
        "poly_min_test_pred": round(float(poly_pred.min()), 2),
    }

    # ---- 5. Ablations -----------------------------------------------------
    ablation = {"Full model (scaled, 64-32)": m_nn}
    _, _, hist_raw, _, m_raw = run_nn(train, val, test, FEATURES, cfg, seed, scale_features=False)
    ablation["No feature scaling"] = m_raw
    plots.scaling_ablation(hist, hist_raw, FIG / "fig11_scaling_ablation.png")
    for f in FEATURES:
        kept = [x for x in FEATURES if x != f]
        ablation[f"Without {plots.LABELS[f].split(' (')[0].lower()}"] = run_nn(train, val, test, kept, cfg, seed)[4]
    for hidden in ([32], [128, 64, 32]):
        ablation[f"Architecture {'-'.join(map(str, hidden))}"] = run_nn(train, val, test, FEATURES, cfg, seed, hidden=hidden)[4]
    out["ablation"] = ablation
    for k, v in ablation.items():
        print(f"[ablation] {k}: R2 {v['r2']:.4f}  MAE {v['mae']:.3f}")

    # ---- 6. 5-fold cross-validation on the development set ----------------
    kf = KFold(cfg["cv_folds"], shuffle=True, random_state=seed)
    cv = {name: [] for name in comparison}
    for fold, (tr_idx, te_idx) in enumerate(kf.split(dev)):
        f_train_all, f_test = dev.iloc[tr_idx], dev.iloc[te_idx]
        f_train, f_val = train_test_split(f_train_all, test_size=0.15, random_state=seed + fold)
        cv["Neural Network (YieldNet)"].append(run_nn(f_train, f_val, f_test, FEATURES, cfg, seed + fold)[4])
        for name, est in baseline_models(cfg, seed).items():
            est.fit(f_train_all[FEATURES], f_train_all[TARGET])
            cv[name].append(regression_metrics(f_test[TARGET], est.predict(f_test[FEATURES])))
    out["cross_validation"] = {
        name: {k: {"mean": round(float(np.mean([r[k] for r in runs])), 4),
                   "std": round(float(np.std([r[k] for r in runs])), 4)} for k in ("r2", "mae", "rmse")}
        for name, runs in cv.items()}
    for name, s in out["cross_validation"].items():
        print(f"[cv] {name}: R2 {s['r2']['mean']:.4f} +/- {s['r2']['std']:.4f}  MAE {s['mae']['mean']:.3f}")

    # ---- 7. Save ----------------------------------------------------------
    out["environment"] = {"torch": torch.__version__, "numpy": np.__version__, "pandas": pd.__version__}
    (ROOT / "results" / "metrics.json").write_text(json.dumps(out, indent=2))
    rows = [{"model": n, **{k: round(v[k], 4) for k in ("r2", "mae", "rmse")}} for n, v in comparison.items()]
    pd.DataFrame(rows).to_csv(ROOT / "results" / "comparison.csv", index=False)
    rows = [{"variant": n, **{k: round(v[k], 4) for k in ("r2", "mae", "rmse")}} for n, v in ablation.items()]
    pd.DataFrame(rows).to_csv(ROOT / "results" / "ablation.csv", index=False)
    print("[done] results/metrics.json written")


if __name__ == "__main__":
    main()
