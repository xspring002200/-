"""
Module 3: 小样本防过拟合与模型训练 (Training & Optimization)

Features
--------
- GPU-aware training loop
- Mini-batch iteration (default batch_size=16)
- Adam optimiser with L2 regularisation (weight_decay)
- Dropout already baked into the model (see model.py)
- Epoch-wise loss reporting
"""

from __future__ import annotations

import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from model import LSTMCNNAttnModel


def train(
    model: LSTMCNNAttnModel,
    X_train: torch.Tensor,
    Y_train: torch.Tensor,
    X_test: torch.Tensor,
    Y_test: torch.Tensor,
    epochs: int = 300,
    batch_size: int = 16,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    print_every: int = 50,
) -> dict:
    """Train the model and return a history dict with train/val loss arrays.

    Parameters
    ----------
    model       : LSTMCNNAttnModel already placed on the correct device
    X_train / Y_train / X_test / Y_test : tensors on the same device
    epochs      : number of full passes over the training data
    batch_size  : mini-batch size (16 is good for ~200 samples)
    lr          : Adam learning rate
    weight_decay: L2 regularisation coefficient in Adam
    print_every : how often to print a progress line
    """
    device = next(model.parameters()).device

    # DataLoader
    train_ds = TensorDataset(X_train, Y_train)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    criterion = nn.MSELoss()
    optimiser = torch.optim.Adam(model.parameters(),
                                 lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=epochs, eta_min=lr * 0.01
    )

    history = {"train_loss": [], "val_loss": []}
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        # ── Training ────────────────────────────────────────────────────────
        model.train()
        running_loss = 0.0
        for X_batch, Y_batch in train_loader:
            optimiser.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, Y_batch)
            loss.backward()
            # Gradient clipping to stabilise small-sample training
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimiser.step()
            running_loss += loss.item() * X_batch.size(0)

        scheduler.step()
        avg_train_loss = running_loss / len(train_ds)

        # ── Validation ──────────────────────────────────────────────────────
        model.eval()
        with torch.no_grad():
            val_preds = model(X_test)
            val_loss = criterion(val_preds, Y_test).item()

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val_loss)

        if epoch % print_every == 0 or epoch == 1:
            elapsed = time.time() - t0
            print(
                f"Epoch [{epoch:4d}/{epochs}]  "
                f"Train Loss: {avg_train_loss:.6f}  "
                f"Val Loss: {val_loss:.6f}  "
                f"Elapsed: {elapsed:.1f}s"
            )

    return history
