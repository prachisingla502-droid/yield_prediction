"""All report figures. One palette, thin marks, recessive axes, print-friendly."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
NEUTRAL = "#f0efec"

LABELS = {
    "temperature_K": "Temperature (K)",
    "pressure_bar": "Pressure (bar)",
    "concentration_molL": "Concentration (mol/L)",
    "yield_pct": "Yield (%)",
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK2,
    "axes.titlesize": 10,
    "axes.titlecolor": INK,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "axes.axisbelow": True,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": INK2,
    "ytick.labelcolor": INK2,
    "legend.frameon": False,
    "figure.dpi": 100,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "figure.facecolor": "white",
})


def _save(fig, path):
    fig.savefig(path)
    plt.close(fig)


def flowchart(path):
    """Two-row snake: steps 1-4 left to right, steps 5-7 right to left underneath."""
    W, H, GAP = 3.4, 1.55, 0.6
    col_x = [0.2 + c * (W + GAP) for c in range(4)]
    row_y = {1: 4.3, 2: 2.0}
    steps = [  # (number, title, body, column, row)
        (1, "Input data", "T, P, C (simulated)\nn = 2,000", 0, 1),
        (2, "Split", "70 / 15 / 15 %\ntrain / val / test", 1, 1),
        (3, "Strict scaling", "StandardScaler fitted\non training set only", 2, 1),
        (4, "YieldNet", "PyTorch 3→64→32→1\nReLU, linear output", 3, 1),
        (5, "Training", "Adam + MSE loss,\nearly stopping", 3, 2),
        (6, "Evaluation", "Test set: R², MAE,\nRMSE, residual plot", 2, 2),
        (7, "Output", "Predicted\nyield (%)", 1, 2),
    ]
    fig, ax = plt.subplots(figsize=(7.2, 3.1))
    ax.set_axis_off()
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 6.0)
    centers = {}
    for n, title, body, c, r in steps:
        x, y = col_x[c], row_y[r]
        accent = BLUE if n in (3, 4) else (AQUA if n == 7 else AXIS)
        ax.add_patch(FancyBboxPatch((x, y), W, H, boxstyle="round,pad=0.02,rounding_size=0.15",
                                    linewidth=1.4, edgecolor=accent, facecolor="white"))
        ax.text(x + W / 2, y + H - 0.32, f"{n}. {title}", ha="center", va="center", fontsize=8.6,
                weight="bold", color=INK)
        ax.text(x + W / 2, y + 0.55, body, ha="center", va="center", fontsize=7.6, color=INK2, linespacing=1.25)
        centers[n] = (x, y)
    arrow = dict(arrowstyle="-|>", mutation_scale=11, color=MUTED, linewidth=1.2)
    for n in (1, 2, 3):   # row 1, left to right
        x, y = centers[n]
        ax.add_patch(FancyArrowPatch((x + W + 0.05, y + H / 2), (x + W + GAP - 0.05, y + H / 2), **arrow))
    x, y = centers[4]     # down from 4 to 5
    ax.add_patch(FancyArrowPatch((x + W / 2, y - 0.05), (x + W / 2, row_y[2] + H + 0.05), **arrow))
    for n in (5, 6):      # row 2, right to left
        x, y = centers[n]
        ax.add_patch(FancyArrowPatch((x - 0.05, y + H / 2), (x - GAP + 0.05, y + H / 2), **arrow))
    # comparison models feed the same evaluation
    bw = W + 2.4
    bx, by = col_x[2] + W / 2 - bw / 2, 0.15
    ax.add_patch(FancyBboxPatch((bx, by), bw, 1.05, boxstyle="round,pad=0.02,rounding_size=0.12",
                                linewidth=1, edgecolor=ORANGE, facecolor="white", linestyle="--"))
    ax.text(bx + bw / 2, by + 0.52, "Baselines on the same split:\nLinear, Poly-2, Random Forest",
            ha="center", va="center", fontsize=7.4, color=INK2, linespacing=1.25)
    ax.add_patch(FancyArrowPatch((col_x[2] + W / 2, by + 1.1), (col_x[2] + W / 2, row_y[2] - 0.05),
                                 linestyle="--", **arrow))
    _save(fig, path)


def loss_and_parity(history, y_true, y_pred, r2, path):
    """Training curve and predicted-vs-actual side by side (compact version for the report)."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), gridspec_kw={"width_ratios": [1.45, 1]})
    ax = axes[0]
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    ax.plot(epochs, history["train_loss"], color=BLUE, linewidth=2, label="Training loss")
    ax.plot(epochs, history["val_loss"], color=ORANGE, linewidth=2, label="Validation loss")
    best = history["best_epoch"] + 1
    ax.axvline(best, color=MUTED, linewidth=1, linestyle="--")
    ax.set_yscale("log")
    ax.text(best, 0.35, f"best epoch = {best}  ", color=INK2, fontsize=8, ha="right")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE (standardised target, log)")
    ax.set_title("(a) Training and validation loss")
    ax.legend(loc="upper center")
    ax = axes[1]
    ax.scatter(y_true, y_pred, s=10, color=BLUE, alpha=0.6, linewidths=0)
    lim = [min(y_true.min(), y_pred.min()) - 2, max(y_true.max(), y_pred.max()) + 2]
    ax.plot(lim, lim, color=INK2, linewidth=1.2, linestyle="--", label="y = x")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect("equal")
    ax.set_xlabel("Actual yield (%)")
    ax.set_ylabel("Predicted yield (%)")
    ax.set_title(f"(b) Predicted vs actual (R² = {r2:.3f})")
    ax.legend(loc="upper left")
    fig.tight_layout()
    _save(fig, path)


def distributions(df, features, target, path):
    cols = features + [target]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2))
    for ax, col in zip(axes.ravel(), cols):
        ax.hist(df[col], bins=30, color=BLUE, edgecolor="white", linewidth=0.8)
        ax.set_xlabel(LABELS[col])
        ax.set_ylabel("Count")
        ax.grid(axis="x", visible=False)
    fig.suptitle("Distribution of inputs and target", fontsize=10, weight="bold", color=INK)
    fig.tight_layout()
    _save(fig, path)


def correlation(df, cols, path):
    corr = df[cols].corr().to_numpy()
    cmap = LinearSegmentedColormap.from_list("div", [ORANGE, NEUTRAL, BLUE])
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    im = ax.imshow(corr, cmap=cmap, vmin=-1, vmax=1)
    names = [LABELS[c].split(" (")[0] for c in cols]
    ax.set_xticks(range(len(cols)), names, rotation=20, ha="right")
    ax.set_yticks(range(len(cols)), names)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    for i in range(len(cols)):
        for j in range(len(cols)):
            v = 0.0 if abs(corr[i, j]) < 0.005 else corr[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", color=INK, fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Pearson r")
    ax.set_title("Linear correlation matrix")
    _save(fig, path)


def yield_vs_features(df, features, target, path):
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.0), sharey=True)
    for ax, col in zip(axes, features):
        ax.scatter(df[col], df[target], s=6, color=BLUE, alpha=0.35, linewidths=0)
        ax.set_xlabel(LABELS[col])
    axes[0].set_ylabel(LABELS[target])
    fig.suptitle("Yield against each reaction condition", fontsize=10, weight="bold", color=INK)
    fig.tight_layout()
    _save(fig, path)


def scaling_effect(X_raw, X_scaled, features, path):
    names = [LABELS[f].split(" (")[0] for f in features]
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.2))
    for ax, X, title in ((axes[0], X_raw, "Before scaling (raw units)"),
                         (axes[1], X_scaled, "After StandardScaler (z-scores)")):
        bp = ax.boxplot([X[:, i] for i in range(X.shape[1])], tick_labels=names, widths=0.5,
                        patch_artist=True, medianprops={"color": INK, "linewidth": 1.2},
                        whiskerprops={"color": MUTED}, capprops={"color": MUTED})
        for patch in bp["boxes"]:
            patch.set(facecolor="#cde2fb", edgecolor=BLUE, linewidth=1)
        ax.set_title(title)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Feature value")
    fig.tight_layout()
    _save(fig, path)


def loss_curves(history, path):
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    ax.plot(epochs, history["train_loss"], color=BLUE, linewidth=2, label="Training loss")
    ax.plot(epochs, history["val_loss"], color=ORANGE, linewidth=2, label="Validation loss")
    best = history["best_epoch"] + 1
    ax.axvline(best, color=MUTED, linewidth=1, linestyle="--")
    ax.set_yscale("log")
    ax.text(best, 0.35, f"best epoch = {best}  ", color=INK2, fontsize=8, ha="right")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE (standardised target, log scale)")
    ax.set_title("Training and validation loss")
    ax.legend(loc="upper center")
    _save(fig, path)


def pred_vs_actual(y_true, y_pred, r2, path):
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    ax.scatter(y_true, y_pred, s=12, color=BLUE, alpha=0.6, linewidths=0)
    lim = [min(y_true.min(), y_pred.min()) - 2, max(y_true.max(), y_pred.max()) + 2]
    ax.plot(lim, lim, color=INK2, linewidth=1.2, linestyle="--", label="Perfect prediction (y = x)")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect("equal")
    ax.set_xlabel("Actual yield (%)")
    ax.set_ylabel("Predicted yield (%)")
    ax.set_title(f"Predicted vs actual (test set, R² = {r2:.3f})")
    ax.legend(loc="upper left")
    _save(fig, path)


def residuals(y_pred, resid, noise_std, path):
    sd = resid.std()
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), gridspec_kw={"width_ratios": [1.6, 1]})
    ax = axes[0]
    ax.axhspan(-2 * sd, 2 * sd, color=NEUTRAL, zorder=0, label=f"±2σ band (σ = {sd:.2f})")
    ax.axhline(0, color=INK2, linewidth=1.2)
    ax.scatter(y_pred, resid, s=12, color=BLUE, alpha=0.6, linewidths=0, zorder=2)
    ax.set_xlabel("Predicted yield (%)")
    ax.set_ylabel("Residual = actual − predicted (%)")
    ax.set_title("Residuals vs predicted")
    ax.legend(loc="upper right")
    ax = axes[1]
    ax.hist(resid, bins=30, color=BLUE, edgecolor="white", linewidth=0.8, density=True)
    xs = np.linspace(resid.min(), resid.max(), 200)
    ax.plot(xs, np.exp(-0.5 * (xs / noise_std) ** 2) / (noise_std * np.sqrt(2 * np.pi)), color=ORANGE,
            linewidth=2, label=f"N(0, {noise_std:g}²) noise")
    ax.set_xlabel("Residual (%)")
    ax.set_ylabel("Density")
    ax.set_title("Residual distribution")
    ax.legend(loc="upper left", fontsize=7.5)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    _save(fig, path)


def residual_contrast(panels, path):
    """panels: list of (title, y_pred, resid, color). Same y-axis so spreads are comparable."""
    fig, axes = plt.subplots(1, len(panels), figsize=(9, 3.3), sharey=True)
    for ax, (title, y_pred, resid, color) in zip(axes, panels):
        ax.axhline(0, color=INK2, linewidth=1.2)
        ax.scatter(y_pred, resid, s=10, color=color, alpha=0.6, linewidths=0)
        ax.set_xlabel("Predicted yield (%)")
        ax.set_title(title, fontsize=9)
    axes[0].set_ylabel("Residual (%)")
    fig.tight_layout()
    _save(fig, path)


def model_comparison(rows, path):
    """rows: list of (name, r2, mae); the neural network row is highlighted."""
    names = [r[0] for r in rows]
    colors = [BLUE if "Neural" in n else "#b9b8b1" for n in names]
    y = np.arange(len(rows))
    fig, axes = plt.subplots(1, 2, figsize=(9, 2.9), sharey=True)
    for ax, idx, title, fmt in ((axes[0], 1, "R² on test set (higher is better)", "{:.3f}"),
                                (axes[1], 2, "MAE on test set, yield % (lower is better)", "{:.2f}")):
        vals = [r[idx] for r in rows]
        ax.barh(y, vals, color=colors, height=0.6)
        for yi, v in zip(y, vals):
            ax.text(v, yi, " " + fmt.format(v), va="center", color=INK, fontsize=8)
        ax.set_title(title)
        ax.grid(axis="y", visible=False)
        ax.set_xlim(0, max(vals) * 1.18)
    axes[0].set_yticks(y, names)
    axes[0].invert_yaxis()
    fig.tight_layout()
    _save(fig, path)


def scaling_ablation(hist_scaled, hist_raw, path):
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    ax.plot(np.arange(1, len(hist_scaled["val_loss"]) + 1), hist_scaled["val_loss"], color=BLUE,
            linewidth=2, label="With feature scaling")
    ax.plot(np.arange(1, len(hist_raw["val_loss"]) + 1), hist_raw["val_loss"], color=ORANGE,
            linewidth=2, label="Without feature scaling (raw units)")
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation MSE (log scale)")
    ax.set_title("Ablation: effect of feature scaling on training")
    ax.legend(loc="upper right")
    _save(fig, path)


def response_curves(curves, path):
    """curves: dict panel -> {"x": array, "xlabel": str, "series": [(label, y, color, style)]}."""
    fig, axes = plt.subplots(1, len(curves), figsize=(9, 3.5), sharey=True)
    for ax, (title, panel) in zip(axes, curves.items()):
        for label, ys, color, style in panel["series"]:
            ax.plot(panel["x"], ys, color=color, linestyle=style, linewidth=2, label=label)
        ax.set_xlabel(panel["xlabel"])
        ax.set_title(title, fontsize=9)
    axes[0].set_ylabel("Yield (%)")
    axes[0].legend(loc="upper left", fontsize=7.5)
    fig.tight_layout()
    _save(fig, path)
