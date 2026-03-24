"""
Module 4: 精度自动评估与发表级图表输出 (Evaluation & Visualization)

Features
--------
- Inverse-transform predictions back to physical units (mg/kg)
- Auto-compute R² and RMSE for each of the 8 heavy metals
- Generate a publication-quality 2×4 scatter-plot figure
  (1:1 reference line, geo-science colour style, per-metal metrics)
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from model import LSTMCNNAttnModel


# ── Colour palette: muted geoscience / cartography style ──────────────────────
_SCATTER_COLORS = [
    "#2E86AB",  # steel-blue   – Cu
    "#A23B72",  # plum         – Zn
    "#F18F01",  # amber        – Pb
    "#4CAF50",  # leaf-green   – Cd
    "#3F51B5",  # indigo       – Cr
    "#00BCD4",  # teal         – Ni
    "#8BC34A",  # light-green  – As
    "#FF7043",  # deep-orange  – Hg
]


def evaluate(
    model: LSTMCNNAttnModel,
    X_test: torch.Tensor,
    Y_test_scaled: torch.Tensor,
    scaler_Y: StandardScaler,
    metal_names: list[str],
) -> dict:
    """Return a dict with 'r2', 'rmse', 'y_true', 'y_pred' (all in mg/kg)."""
    model.eval()
    with torch.no_grad():
        Y_pred_scaled = model(X_test).cpu().numpy()

    Y_test_np = Y_test_scaled.cpu().numpy()

    # Inverse-transform to physical units
    Y_true = scaler_Y.inverse_transform(Y_test_np)
    Y_pred = scaler_Y.inverse_transform(Y_pred_scaled)

    r2_list, rmse_list = [], []
    for i in range(len(metal_names)):
        r2 = r2_score(Y_true[:, i], Y_pred[:, i])
        rmse = np.sqrt(mean_squared_error(Y_true[:, i], Y_pred[:, i]))
        r2_list.append(r2)
        rmse_list.append(rmse)
        print(f"  {metal_names[i]:3s}  R²={r2:.4f}  RMSE={rmse:.4f} mg/kg")

    return {
        "r2": r2_list,
        "rmse": rmse_list,
        "y_true": Y_true,
        "y_pred": Y_pred,
    }


def plot_results(
    metrics: dict,
    metal_names: list[str],
    save_path: str = "heavy_metal_prediction.png",
) -> None:
    """Draw a 2×4 scatter-plot figure and save to *save_path*.

    Each subplot shows:
    - Measured (x-axis) vs Predicted (y-axis) values in mg/kg
    - A 1:1 reference line (perfect prediction)
    - R² and RMSE in the title
    """
    Y_true: np.ndarray = metrics["y_true"]
    Y_pred: np.ndarray = metrics["y_pred"]
    r2_list: list[float] = metrics["r2"]
    rmse_list: list[float] = metrics["rmse"]

    fig, axes = plt.subplots(2, 4, figsize=(18, 9), dpi=150)
    fig.suptitle(
        "Hyperspectral Inversion of Soil Heavy Metals\n"
        "(LSTM-CNN-Attention Multi-Task Model)",
        fontsize=14, fontweight="bold", y=1.02,
    )

    for idx, (ax, name) in enumerate(zip(axes.flat, metal_names)):
        y_true_i = Y_true[:, idx]
        y_pred_i = Y_pred[:, idx]
        color = _SCATTER_COLORS[idx % len(_SCATTER_COLORS)]

        # 1:1 line range
        lo = min(y_true_i.min(), y_pred_i.min()) * 0.95
        hi = max(y_true_i.max(), y_pred_i.max()) * 1.05

        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1.2,
                label="1:1 Line", zorder=1)
        ax.scatter(y_true_i, y_pred_i, c=color, edgecolors="white",
                   s=60, alpha=0.85, zorder=2)

        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel("Measured (mg/kg)", fontsize=9)
        ax.set_ylabel("Predicted (mg/kg)", fontsize=9)
        ax.set_title(
            f"{name}\nR²={r2_list[idx]:.4f}   RMSE={rmse_list[idx]:.4f}",
            fontsize=10, fontweight="bold",
        )
        ax.tick_params(labelsize=8)
        ax.legend(fontsize=7, loc="upper left")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, linestyle=":", alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure saved to: {os.path.abspath(save_path)}")
