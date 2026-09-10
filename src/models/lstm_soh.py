"""
Recurrent models for SOH prediction from a window of past cycle summary
features, conditioned on static cell metadata (C-rate, temperature, form
factor).

Four interchangeable recurrent backbones are offered — `lstm` (the default
and the one the shipped production model uses), `gru`, `rnn` (vanilla
Elman) and `bilstm` (bidirectional LSTM) — all sharing the same residual
"predict the correction to the last observed SOH" structure described in
SOHRecurrent's docstring, and all trained/evaluated identically so their
LOCO numbers are directly comparable.

Kept deliberately small: with only a handful of cells and ~100 cycles each,
a large model would memorize rather than generalize. Regularization
(dropout, weight decay, early stopping on the held-out cell) matters more
than capacity here. Note the bidirectional variant is included for
comparison completeness, but reading the window backwards has no causal
justification for a forecasting task — it only ever sees a closed history
window, never future cycles, so it is not "cheating", it simply has no
particular reason to help.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

# Selectable recurrent backbones. Keys are the identifiers used on the CLI,
# in the saved checkpoints, and in the web app's model picker.
ARCH_CHOICES = ("lstm", "gru", "rnn", "bilstm")

ARCH_LABELS = {
    "lstm": "LSTM",
    "gru": "GRU",
    "rnn": "Simple RNN",
    "bilstm": "Bi-LSTM",
}


@dataclass
class LSTMConfig:
    n_features: int
    n_static: int
    soh_feature_idx: int = 0  # index of soh_pct within the feature vector
    hidden_size: int = 16
    num_layers: int = 1
    dropout: float = 0.3
    static_embed_size: int = 8
    arch: str = "lstm"  # one of ARCH_CHOICES


class SOHRecurrent(nn.Module):
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

    The recurrent backbone is selected by `cfg.arch` (see ARCH_CHOICES).
    Every variant keeps the same static-metadata embedding, the same
    correction head, and the same zero-initialized final layer, so the only
    thing that differs between reported architectures is how the window is
    encoded — which is what makes their LOCO scores comparable.
    """

    def __init__(self, cfg: LSTMConfig):
        super().__init__()
        self.cfg = cfg
        arch = getattr(cfg, "arch", "lstm")
        if arch not in ARCH_CHOICES:
            raise ValueError(f"Unknown arch {arch!r}; expected one of {ARCH_CHOICES}")
        self.arch = arch

        self.static_embed = nn.Sequential(
            nn.Linear(cfg.n_static, cfg.static_embed_size),
            nn.ReLU(),
        )

        rnn_kwargs = dict(
            input_size=cfg.n_features,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            batch_first=True,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
        )
        if arch == "lstm":
            self.rnn = nn.LSTM(**rnn_kwargs)
        elif arch == "bilstm":
            self.rnn = nn.LSTM(bidirectional=True, **rnn_kwargs)
        elif arch == "gru":
            self.rnn = nn.GRU(**rnn_kwargs)
        elif arch == "rnn":
            self.rnn = nn.RNN(nonlinearity="tanh", **rnn_kwargs)

        # A bidirectional encoder concatenates the final forward and final
        # backward hidden states, so the head sees twice the width.
        encoder_out = cfg.hidden_size * (2 if arch == "bilstm" else 1)

        self.head = nn.Sequential(
            nn.Linear(encoder_out + cfg.static_embed_size, 16),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(16, 1),
        )
        # Initialize the correction head to output ~0 at the start of
        # training, so the model begins as pure persistence and only
        # deviates from it where the data supports doing so.
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def _encode(self, x_seq: torch.Tensor) -> torch.Tensor:
        """Run the recurrent backbone and return the window encoding (B, E)."""
        if self.arch in ("lstm", "bilstm"):
            _, (h_n, _) = self.rnn(x_seq)
        else:  # gru / rnn return only h_n
            _, h_n = self.rnn(x_seq)

        if self.arch == "bilstm":
            # h_n is (num_layers*2, B, H); the last two entries are the final
            # layer's forward and backward states respectively.
            return torch.cat([h_n[-2], h_n[-1]], dim=1)  # (B, 2H)
        return h_n[-1]  # (B, H)

    def forward(
        self,
        x_seq: torch.Tensor,
        x_static: torch.Tensor,
        raw_last_soh: torch.Tensor,
    ) -> torch.Tensor:
        # x_seq: (B, T, n_features) NORMALIZED features (model input)
        # x_static: (B, n_static) NORMALIZED static features
        # raw_last_soh: (B,) UN-normalized last observed SOH%, for the skip connection
        h_last = self._encode(x_seq)
        static_emb = self.static_embed(x_static)  # (B, static_embed_size)
        combined = torch.cat([h_last, static_emb], dim=1)
        correction = self.head(combined).squeeze(-1)  # (B,)
        return raw_last_soh + correction


# Backwards-compatible alias: the class was named SOHLSTM when only the LSTM
# backbone existed, and that name is what older checkpoints/imports use.
SOHLSTM = SOHRecurrent
