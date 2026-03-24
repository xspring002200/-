"""
Main entry-point: Soil Heavy Metal Hyperspectral Inversion
==========================================================

Orchestrates all four pipeline modules in sequence:

  Module 1 – data_preprocessing.py  : load, denoise, normalise, split
  Module 2 – model.py               : LSTM-CNN-Attention architecture
  Module 3 – train.py               : GPU-aware training with L2 + Dropout
  Module 4 – evaluate.py            : R²/RMSE metrics + 2×4 scatter figure

Usage
-----
  # Generate synthetic sample data (first run only):
  python generate_sample_data.py

  # Run the full pipeline:
  python main.py

  # Override defaults via CLI flags:
  python main.py --data my_data.csv --epochs 500 --batch_size 16
"""

from __future__ import annotations

import argparse
import os

import torch

from data_preprocessing import preprocess
from evaluate import evaluate, plot_results
from model import build_model
from train import train


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Soil Heavy Metal Hyperspectral Inversion — LSTM-CNN-Attention"
    )
    p.add_argument("--data",        default="my_data.csv",
                   help="Path to the CSV data file")
    p.add_argument("--n_metals",    type=int, default=8,
                   help="Number of heavy-metal columns at the END of the CSV "
                        "(default 8). All preceding numeric columns are treated "
                        "as spectral bands.")
    p.add_argument("--epochs",      type=int, default=300,
                   help="Number of training epochs (200–500 recommended)")
    p.add_argument("--batch_size",  type=int, default=16,
                   help="Mini-batch size")
    p.add_argument("--lr",          type=float, default=1e-3,
                   help="Adam learning rate")
    p.add_argument("--weight_decay",type=float, default=1e-4,
                   help="L2 regularisation weight-decay in Adam")
    p.add_argument("--dropout",     type=float, default=0.3,
                   help="Dropout probability in the model")
    p.add_argument("--sg_window",   type=int, default=11,
                   help="Savitzky-Golay window length (odd integer)")
    p.add_argument("--sg_poly",     type=int, default=3,
                   help="Savitzky-Golay polynomial order")
    p.add_argument("--sg_deriv",    type=int, default=0,
                   help="SG derivative order (0=smooth only, 1=1st derivative)")
    p.add_argument("--output",      default="heavy_metal_prediction.png",
                   help="Output figure path")
    p.add_argument("--model_save",  default="best_model.pth",
                   help="Path to save the trained model weights")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ──────────────────────────────────────────────────────────────────────────
    # Module 1: Data Preprocessing
    # ──────────────────────────────────────────────────────────────────────────
    print("=" * 60)
    print("MODULE 1 — Data Preprocessing")
    print("=" * 60)
    if not os.path.isfile(args.data):
        raise FileNotFoundError(
            f"Data file '{args.data}' not found.\n"
            "Run  python generate_sample_data.py  to create synthetic data."
        )

    data = preprocess(
        filepath=args.data,
        n_metals=args.n_metals,
        sg_window=args.sg_window,
        sg_poly=args.sg_poly,
        sg_deriv=args.sg_deriv,
    )
    device = data["device"]
    print(f"  Device          : {device}")
    print(f"  Training samples: {data['X_train'].shape[0]}")
    print(f"  Test samples    : {data['X_test'].shape[0]}")
    print(f"  Spectral bands  : {data['n_bands']}")
    print(f"  Heavy metals    : {data['metal_names']}")

    # ──────────────────────────────────────────────────────────────────────────
    # Module 2: Build Model
    # ──────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("MODULE 2 — Model Architecture (LSTM-CNN-Attention)")
    print("=" * 60)
    model = build_model(
        n_bands=data["n_bands"],
        n_metals=len(data["metal_names"]),
        dropout=args.dropout,
        device=device,
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total trainable parameters: {n_params:,}")
    print(model)

    # ──────────────────────────────────────────────────────────────────────────
    # Module 3: Train
    # ──────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("MODULE 3 — Training & Optimisation")
    print("=" * 60)
    history = train(
        model=model,
        X_train=data["X_train"],
        Y_train=data["Y_train"],
        X_test=data["X_test"],
        Y_test=data["Y_test"],
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Save model weights
    torch.save(model.state_dict(), args.model_save)
    print(f"\nModel weights saved to: {os.path.abspath(args.model_save)}")

    # ──────────────────────────────────────────────────────────────────────────
    # Module 4: Evaluate & Visualise
    # ──────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("MODULE 4 — Evaluation & Visualisation")
    print("=" * 60)
    metrics = evaluate(
        model=model,
        X_test=data["X_test"],
        Y_test_scaled=data["Y_test"],
        scaler_Y=data["scaler_Y"],
        metal_names=data["metal_names"],
    )
    plot_results(metrics, metal_names=data["metal_names"],
                 save_path=args.output)

    print("\n✓ Pipeline complete.")


if __name__ == "__main__":
    main()
