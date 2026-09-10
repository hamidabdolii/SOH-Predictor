"""
Build sliding-window sequence samples for SOH prediction from the VL npz
caches (and, once available, NASA/CALCE/Oxford caches via an adapter).

Design rationale
-----------------
With only 4-6 cells x ~100 cycles each, a per-cell LSTM would overfit badly.
Instead we build a *pooled* dataset: every cell contributes many overlapping
windows of W consecutive cycles, and the model is trained across all cells
at once, conditioned on static metadata (C-rate, temperature) so a single
model covers every test condition. Evaluation uses leave-one-cell-out (LOCO)
cross-validation, which is the only honest way to estimate generalization
to a battery the model has never seen, given how few cells exist.

Feature normalization: capacity-derived features are expressed as SOH% (already
normalized per-cell by that cell's own C_ref), so raw capacity scale
differences between VL1220 and VL2020 don't leak in directly. Duration and
temperature features are z-scored using statistics computed on the training
folds only (never on the held-out cell), to avoid leakage.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Per-cycle scalar feature keys pulled directly from the npz cache, used as
# the LSTM's per-timestep input (besides soh_pct itself, which is both an
# input feature at past timesteps and the prediction target at the horizon).
SEQUENCE_FEATURE_KEYS = [
    "soh_pct",
    "coulombic_efficiency",
    "dchg_duration_s",
    "cc_chg_duration_s",
    "cv_chg_duration_s",
    "mean_temperature_c",
    "max_temperature_c",
    "min_dchg_voltage_v",
    "max_chg_voltage_v",
    "cv_chg_end_current_a",
]

# Static (per-cell, constant across the window) conditioning features.
C_RATE_VALUES = {"0.05C": 0.05, "0.1C": 0.1, "0.2C": 0.2}
TEMP_VALUES = {"RT": 25.0, "40": 40.0}  # nominal degC; RT approximated as 25


@dataclass
class CellRecord:
    cell_id: str
    form_factor: str
    c_rate: str
    temp_condition: str
    cycles: np.ndarray
    features: dict[str, np.ndarray]  # key -> (n_cycles,) array, keys = SEQUENCE_FEATURE_KEYS

    def static_vector(self) -> np.ndarray:
        c_rate_val = C_RATE_VALUES[self.c_rate] if self.c_rate in C_RATE_VALUES else float(self.c_rate.rstrip("C"))
        temp_val = TEMP_VALUES[self.temp_condition] if self.temp_condition in TEMP_VALUES else float(self.temp_condition)
        form_factor_flag = 1.0 if self.form_factor == "VL2020" else 0.0
        return np.array([c_rate_val, temp_val, form_factor_flag], dtype=np.float32)


def load_cell_record(npz_path: str) -> CellRecord:
    d = np.load(npz_path)
    features = {k: d[k].astype(np.float32) for k in SEQUENCE_FEATURE_KEYS}
    return CellRecord(
        cell_id=str(d["cell_id"]),
        form_factor=str(d["form_factor"]),
        c_rate=str(d["c_rate"]),
        temp_condition=str(d["temp_condition"]),
        cycles=d["cycles"],
        features=features,
    )


@dataclass
class WindowSample:
    cell_id: str
    x_seq: np.ndarray     # (window, n_features) — history window
    x_static: np.ndarray  # (n_static,)
    y: float              # target SOH% at cycle (window_end + horizon)
    target_cycle: int


def make_windows(
    record: CellRecord,
    window: int = 10,
    horizon: int = 1,
    stride: int = 1,
) -> list[WindowSample]:
    n = len(record.cycles)
    feats = np.stack([record.features[k] for k in SEQUENCE_FEATURE_KEYS], axis=1)  # (n, n_features)
    static = record.static_vector()
    samples = []
    for start in range(0, n - window - horizon + 1, stride):
        end = start + window
        target_idx = end + horizon - 1
        if target_idx >= n:
            break
        x_seq = feats[start:end]
        y = float(record.features["soh_pct"][target_idx])
        samples.append(WindowSample(
            cell_id=record.cell_id,
            x_seq=x_seq,
            x_static=static,
            y=y,
            target_cycle=int(record.cycles[target_idx]),
        ))
    return samples


@dataclass
class Normalizer:
    mean: np.ndarray
    std: np.ndarray

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std

    @classmethod
    def fit(cls, samples: list[WindowSample]) -> "Normalizer":
        all_seq = np.concatenate([s.x_seq for s in samples], axis=0)  # (N, n_features)
        mean = all_seq.mean(axis=0)
        std = all_seq.std(axis=0)
        std[std < 1e-6] = 1.0
        return cls(mean=mean, std=std)


@dataclass
class StaticNormalizer:
    mean: np.ndarray
    std: np.ndarray

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std

    @classmethod
    def fit(cls, samples: list[WindowSample]) -> "StaticNormalizer":
        all_static = np.stack([s.x_static for s in samples], axis=0)
        mean = all_static.mean(axis=0)
        std = all_static.std(axis=0)
        std[std < 1e-6] = 1.0
        return cls(mean=mean, std=std)
