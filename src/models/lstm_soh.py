"""
LSTM model for SOH prediction from a window of past cycle summary features,
conditioned on static cell metadata (C-rate, temperature, form factor).

Kept deliberately small: with only a handful of cells and ~100 cycles each,
a large model would memorize rather than generalize. Regularization
(dropout, weight decay, early stopping on the held-out cell) matters more
than capacity here.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class LSTMConfig:
    n_features: int
    n_static: int
    soh_feature_idx: int = 0  # index of soh_pct within the feature vector
    hidden_size: int = 16
    num_layers: int = 1
    dropout: float = 0.3
    static_embed_size: int = 8


class SOHLSTM(nn.Module):
    """Predicts SOH at the target cycle as (last observed SOH) + (learned
    correction). The residual/skip connection is what makes this generalize
    across cells with a handful of training batteries: without it, the
    network must learn the *absolute* SOH scale from scratch per cell and
    badly overfits to the training cells' specific trajectories (verified
    empirically: plain seq2one LSTM here scored far worse than a naive
    persistence baseline under leave-one-cell-out evaluation). With the
    residual, the network only needs to learn the *change* in SOH, a much
    smaller and more transferable quantity, and the raw feature values
    (which include noise/interpolation from the underlying capacity signal)
    only refine that correction.
    """

    def __init__(self, cfg: LSTMConfig):
        super().__init__()
        self.cfg = cfg
        self.static_embed = nn.Sequential(
            nn.Linear(cfg.n_static, cfg.static_embed_size),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(
            input_size=cfg.n_features,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            batch_first=True,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(cfg.hidden_size + cfg.static_embed_size, 16),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(16, 1),
        )
        # Initialize the correction head to output ~0 at the start of
        # training, so the model begins as pure persistence and only
        # deviates from it where the data supports doing so.
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(
        self,
        x_seq: torch.Tensor,
        x_static: torch.Tensor,
        raw_last_soh: torch.Tensor,
    ) -> torch.Tensor:
        # x_seq: (B, T, n_features) NORMALIZED features (model input)
        # x_static: (B, n_static) NORMALIZED static features
        # raw_last_soh: (B,) UN-normalized last observed SOH%, for the skip connection
        _, (h_n, _) = self.lstm(x_seq)
        h_last = h_n[-1]  # (B, hidden_size)
        static_emb = self.static_embed(x_static)  # (B, static_embed_size)
        combined = torch.cat([h_last, static_emb], dim=1)
        correction = self.head(combined).squeeze(-1)  # (B,)
        return raw_last_soh + correction
