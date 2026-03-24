"""
Module 1: 测绘级数据自动化接入与预处理 (Data Preprocessing)

Supports:
- Reading my_data.csv and splitting into spectral features (X) and
  heavy-metal labels (Y)
- CSV format: all columns except the last N_METALS columns are spectral
  bands; the last N_METALS columns are heavy-metal concentrations.
  Column names are read directly from the CSV header.
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

# Default number of heavy-metal output columns (last N columns of the CSV)
N_METALS = 8


def load_csv(
    filepath: str = "my_data.csv",
    n_metals: int = N_METALS,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Read *my_data.csv* and return ``(X, Y, metal_names)``.

    CSV format expected
    -------------------
    - Every row is one sampling point (~200 rows is fine).
    - All columns **except** the last *n_metals* columns are spectral-band
      values (e.g. band_1, band_2, … band_N).
    - The **last** *n_metals* columns are heavy-metal concentrations
      (e.g. Cd, Cu, Pb, Zn, Cr, Ni, As, Hg).  Column names are taken
      directly from the CSV header so no renaming is required.
    - Non-numeric columns (e.g. sample-ID strings) are dropped automatically.

    Returns
    -------
    X : np.ndarray, shape (n_samples, n_bands)   – spectral features
    Y : np.ndarray, shape (n_samples, n_metals)  – metal concentrations
    metal_names : list[str]                      – column names of the metals
    """
    df = pd.read_csv(filepath)
    # Drop any non-numeric columns (e.g. sample IDs)
    df = df.select_dtypes(include=[np.number])

    if df.shape[1] <= n_metals:
        raise ValueError(
            f"CSV must have more than {n_metals} numeric columns "
            f"(got {df.shape[1]}).  Check that all band columns are numeric "
            f"and that n_metals ({n_metals}) is correct."
        )

    metal_names = list(df.columns[-n_metals:])
    X = df.iloc[:, :-n_metals].values.astype(np.float32)
    Y = df.iloc[:, -n_metals:].values.astype(np.float32)
    return X, Y, metal_names


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
    n_metals: int = N_METALS,
    sg_window: int = 11,
    sg_poly: int = 3,
    sg_deriv: int = 0,
    test_size: float = 0.2,
    random_state: int = 42,
    device: str | None = None,
) -> dict:
    """Full preprocessing pipeline.

    Parameters
    ----------
    filepath   : path to the CSV file.
    n_metals   : number of heavy-metal columns at the **end** of the CSV
                 (default 8).  All preceding numeric columns are treated as
                 spectral bands.
    sg_window  : Savitzky-Golay window length (odd integer).
    sg_poly    : Savitzky-Golay polynomial order.
    sg_deriv   : derivative order (0 = smooth only, 1 = 1st derivative).
    test_size  : fraction of samples used for testing (default 0.20).
    random_state : random seed for reproducible splits.
    device     : ``"cuda"`` or ``"cpu"``; auto-detected when *None*.

    Returns a dict with keys:
        X_train, X_test, Y_train, Y_test  – PyTorch tensors (on *device*)
        scaler_X, scaler_Y                 – fitted StandardScaler objects
        metal_names                        – list[str] from CSV column headers
        device                             – torch.device in use
        n_bands                            – number of spectral bands
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    # 1. Load data – metal names are read from the CSV header
    X_raw, Y_raw, metal_names = load_csv(filepath, n_metals=n_metals)

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
        "metal_names": metal_names,
        "device": dev,
        "n_bands": X_scaled.shape[1],
    }
