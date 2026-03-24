"""
Helper script: generate synthetic my_data.csv for testing.

Produces 200 samples with:
  - 150 simulated spectral-reflectance bands (correlated Gaussian)
  - 8 heavy-metal concentrations (Cu, Zn, Pb, Cd, Cr, Ni, As, Hg)
    loosely correlated with a few spectral features to give the model
    something real to learn.

Run once before executing main.py:
    python generate_sample_data.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

RNG = np.random.default_rng(2024)

N_SAMPLES = 200
N_BANDS = 150

# Realistic approximate ranges for mine-area soil (mg/kg)
METAL_PARAMS = {
    "Cu":  (35.0, 15.0),
    "Zn":  (130.0, 40.0),
    "Pb":  (55.0, 20.0),
    "Cd":  (0.45, 0.15),
    "Cr":  (70.0, 18.0),
    "Ni":  (28.0, 8.0),
    "As":  (12.0, 4.0),
    "Hg":  (0.08, 0.03),
}


def generate() -> None:
    # ── Spectral reflectance matrix ───────────────────────────────────────────
    # Simulate correlated bands via a Gaussian random field
    band_idx = np.arange(N_BANDS)
    # Correlation decays exponentially with band distance
    cov = np.exp(-0.5 * (band_idx[:, None] - band_idx[None, :]) ** 2 / 400.0)
    cov += np.eye(N_BANDS) * 1e-6

    # Base spectral curves
    X = RNG.multivariate_normal(
        mean=0.12 + 0.08 * np.sin(band_idx / N_BANDS * np.pi),
        cov=cov * 0.002,
        size=N_SAMPLES,
    ).astype(np.float32)
    X = np.clip(X, 0.01, 0.99)  # keep reflectance in [0, 1]

    # ── Heavy-metal concentrations ────────────────────────────────────────────
    # Use a few "indicator" bands as latent variables
    factor1 = X[:, 30]   # e.g. related to iron-oxide absorption
    factor2 = X[:, 80]   # e.g. clay mineral feature
    factor3 = X[:, 120]  # longer-wave feature

    metals = {}
    for i, (name, (mu, sigma)) in enumerate(METAL_PARAMS.items()):
        # Mix of latent factors + noise to simulate realistic correlation
        noise = RNG.normal(0, sigma * 0.5, N_SAMPLES)
        if i % 3 == 0:
            metal = mu + sigma * (factor1 - factor1.mean()) / factor1.std() + noise
        elif i % 3 == 1:
            metal = mu + sigma * (factor2 - factor2.mean()) / factor2.std() + noise
        else:
            metal = mu + sigma * (factor3 - factor3.mean()) / factor3.std() + noise
        metals[name] = np.clip(metal, mu * 0.1, mu * 5.0)

    # ── Assemble DataFrame ────────────────────────────────────────────────────
    band_cols = [f"band_{i + 1}" for i in range(N_BANDS)]
    df_X = pd.DataFrame(X, columns=band_cols)
    df_Y = pd.DataFrame(metals)
    df = pd.concat([df_X, df_Y], axis=1)

    df.to_csv("my_data.csv", index=False)
    print(f"Saved my_data.csv  ({N_SAMPLES} samples, "
          f"{N_BANDS} spectral bands + {len(METAL_PARAMS)} metals)")


if __name__ == "__main__":
    generate()
