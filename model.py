"""
Module 2: 深度学习特征提取网络构建 (Model Architecture)

Architecture: LSTM → Attention → 1D-CNN → Shared-FC (8 outputs)

Components
----------
LSTMLayer   : extracts sequential spectral dependencies between bands
AttentionLayer : assigns per-timestep importance weights
Conv1DBlock : extracts local spatial features and compresses dimensionality
LSTMCNNAttnModel : full multi-task model predicting 8 heavy metals at once
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionLayer(nn.Module):
    """Scaled dot-product self-attention over the LSTM hidden-state sequence."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.attn_fc = nn.Linear(hidden_dim, 1)

    def forward(self, lstm_out: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        lstm_out : (batch, seq_len, hidden_dim)

        Returns
        -------
        context : (batch, hidden_dim)  – attention-weighted sum
        """
        # score: (batch, seq_len, 1)
        score = self.attn_fc(lstm_out)
        weight = F.softmax(score, dim=1)          # (batch, seq_len, 1)
        context = (weight * lstm_out).sum(dim=1)  # (batch, hidden_dim)
        return context


class LSTMCNNAttnModel(nn.Module):
    """
    Cascade neural network for multi-task heavy-metal prediction.

    Input  : (batch, n_bands)          – standardised spectral reflectance
    Output : (batch, n_metals)         – predicted concentrations (scaled)

    Pipeline
    --------
    1. Reshape to (batch, n_bands, 1) and feed into bidirectional LSTM
       treating each band as one time-step.
    2. Attention layer weights the LSTM hidden states.
    3. 1-D CNN block extracts local spectral features from the LSTM sequence.
    4. Shared fully-connected layers produce 8 metal predictions.
    """

    def __init__(
        self,
        n_bands: int,
        n_metals: int = 8,
        lstm_hidden: int = 64,
        lstm_layers: int = 2,
        cnn_channels: int = 64,
        cnn_kernel: int = 3,
        fc_hidden: int = 128,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        self.n_bands = n_bands

        # ── LSTM ──────────────────────────────────────────────────────────────
        # Input: (batch, n_bands, 1)  →  out: (batch, n_bands, lstm_hidden*2)
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        lstm_out_dim = lstm_hidden * 2  # bidirectional

        # ── Attention ─────────────────────────────────────────────────────────
        self.attention = AttentionLayer(lstm_out_dim)

        # ── 1-D CNN ───────────────────────────────────────────────────────────
        # Operates on LSTM output sequence: (batch, lstm_out_dim, n_bands)
        self.conv1 = nn.Conv1d(lstm_out_dim, cnn_channels,
                               kernel_size=cnn_kernel, padding=cnn_kernel // 2)
        self.conv2 = nn.Conv1d(cnn_channels, cnn_channels // 2,
                               kernel_size=cnn_kernel, padding=cnn_kernel // 2)
        self.pool = nn.AdaptiveAvgPool1d(1)  # → (batch, cnn_channels//2, 1)
        self.bn1 = nn.BatchNorm1d(cnn_channels)
        self.bn2 = nn.BatchNorm1d(cnn_channels // 2)

        # CNN feature dimension
        cnn_feat_dim = cnn_channels // 2

        # ── Shared FC ─────────────────────────────────────────────────────────
        # Concatenate attention context + CNN features
        fc_in = lstm_out_dim + cnn_feat_dim
        self.fc1 = nn.Linear(fc_in, fc_hidden)
        self.fc2 = nn.Linear(fc_hidden, fc_hidden // 2)
        self.out  = nn.Linear(fc_hidden // 2, n_metals)

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    # ── forward ──────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (batch, n_bands)

        Returns
        -------
        (batch, n_metals)
        """
        # Reshape for LSTM: treat each band as a time-step with 1 feature
        x_seq = x.unsqueeze(-1)                   # (batch, n_bands, 1)

        # LSTM
        lstm_out, _ = self.lstm(x_seq)            # (batch, n_bands, hidden*2)

        # Attention
        attn_ctx = self.attention(lstm_out)        # (batch, hidden*2)

        # 1-D CNN  (needs channels-first: batch, channels, length)
        cnn_in = lstm_out.permute(0, 2, 1)        # (batch, hidden*2, n_bands)
        cnn_feat = self.relu(self.bn1(self.conv1(cnn_in)))
        cnn_feat = self.dropout(cnn_feat)
        cnn_feat = self.relu(self.bn2(self.conv2(cnn_feat)))
        cnn_feat = self.pool(cnn_feat).squeeze(-1) # (batch, cnn_channels//2)

        # Concatenate and feed to shared FC
        combined = torch.cat([attn_ctx, cnn_feat], dim=1)
        out = self.relu(self.fc1(combined))
        out = self.dropout(out)
        out = self.relu(self.fc2(out))
        out = self.dropout(out)
        out = self.out(out)                        # (batch, n_metals)
        return out


def build_model(
    n_bands: int,
    n_metals: int = 8,
    lstm_hidden: int = 64,
    lstm_layers: int = 2,
    cnn_channels: int = 64,
    cnn_kernel: int = 3,
    fc_hidden: int = 128,
    dropout: float = 0.3,
    device: str | torch.device = "cpu",
) -> LSTMCNNAttnModel:
    """Instantiate the model and move it to *device*."""
    model = LSTMCNNAttnModel(
        n_bands=n_bands,
        n_metals=n_metals,
        lstm_hidden=lstm_hidden,
        lstm_layers=lstm_layers,
        cnn_channels=cnn_channels,
        cnn_kernel=cnn_kernel,
        fc_hidden=fc_hidden,
        dropout=dropout,
    )
    return model.to(device)
