"""
Module 1: 测绘级数据自动化接入与预处理 (Data Preprocessing)

Supports:
- Reading my_data.csv and splitting into spectral features (X) and
  heavy-metal labels (Y)
- Savitzky-Golay smoothing (and optional first-derivative transform)
- StandardScaler normalisation
- 80/20 train-test split
- PyTorch tensor packaging with GPU support
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from scipy.signal import savgol_filter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Names of the 8 heavy metals (last 8 columns of my_data.csv)
METAL_NAMES = ["Cu", "Zn", "Pb", "Cd", "Cr", "Ni", "As", "Hg"]


def load_csv(filepath: str = "my_data.csv") -> tuple[np.ndarray, np.ndarray]:
    """Read *my_data.csv* and return (X, Y) as numpy arrays.

    The file is expected to have **spectral band columns first**, followed by
    exactly 8 heavy-metal concentration columns (Cu, Zn, Pb, Cd, Cr, Ni, As,
    Hg).  All columns must be numeric.
    """
    df = pd.read_csv(filepath)
    # Drop any non-numeric columns (e.g. sample IDs)
    df = df.select_dtypes(include=[np.number])

    n_metals = len(METAL_NAMES)
    if df.shape[1] <= n_metals:
        raise ValueError(
            f"CSV must have more than {n_metals} numeric columns "
            f"(got {df.shape[1]})."
        )

    X = df.iloc[:, :-n_metals].values.astype(np.float32)
    Y = df.iloc[:, -n_metals:].values.astype(np.float32)
    return X, Y


def apply_savgol(
    X: np.ndarray,
    window_length: int = 11,
    polyorder: int = 3,
    deriv: int = 0,
) -> np.ndarray:
    """Apply Savitzky-Golay smoothing (and optional first-derivative).

    Parameters
    ----------
    X:
        Spectral matrix, shape (n_samples, n_bands).
    window_length:
        Length of the filter window (must be odd and > polyorder).
    polyorder:
        Polynomial order.
    deriv:
        Derivative order.  0 = smoothing only; 1 = first derivative.
    """
    # Ensure window_length is odd and large enough for the polynomial order
    if window_length % 2 == 0:
        window_length += 1
    min_window = polyorder + 2
    if min_window % 2 == 0:
        min_window += 1
    window_length = max(window_length, min_window)

    return savgol_filter(X, window_length=window_length, polyorder=polyorder,
                         deriv=deriv, axis=1).astype(np.float32)


def preprocess(
    filepath: str = "my_data.csv",
    sg_window: int = 11,
    sg_poly: int = 3,
    sg_deriv: int = 0,
    test_size: float = 0.2,
    random_state: int = 42,
    device: str | None = None,
) -> dict:
    """Full preprocessing pipeline.

    Returns a dict with keys:
        X_train, X_test, Y_train, Y_test  – PyTorch tensors (on *device*)
        scaler_X, scaler_Y                 – fitted StandardScaler objects
        metal_names                        – list[str]
        device                             – torch.device in use
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    # 1. Load data
    X_raw, Y_raw = load_csv(filepath)

    # 2. Savitzky-Golay smoothing / derivative
    X_sg = apply_savgol(X_raw, window_length=sg_window,
                        polyorder=sg_poly, deriv=sg_deriv)

    # 3. Standardise
    scaler_X = StandardScaler()
    scaler_Y = StandardScaler()
    X_scaled = scaler_X.fit_transform(X_sg).astype(np.float32)
    Y_scaled = scaler_Y.fit_transform(Y_raw).astype(np.float32)

    # 4. Train/test split
    X_tr, X_te, Y_tr, Y_te = train_test_split(
        X_scaled, Y_scaled, test_size=test_size,
        random_state=random_state
    )

    # 5. Convert to tensors
    def to_tensor(arr: np.ndarray) -> torch.Tensor:
        return torch.tensor(arr, dtype=torch.float32).to(dev)

    return {
        "X_train": to_tensor(X_tr),
        "X_test": to_tensor(X_te),
        "Y_train": to_tensor(Y_tr),
        "Y_test": to_tensor(Y_te),
        "scaler_X": scaler_X,
        "scaler_Y": scaler_Y,
        "metal_names": METAL_NAMES,
        "device": dev,
        "n_bands": X_scaled.shape[1],
    }
