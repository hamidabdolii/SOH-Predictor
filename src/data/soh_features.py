"""
Per-cycle SOH labeling and feature extraction for VL coin-cell data.

Two output views are produced per cell, matching what SOH-prediction
literature (and presumably the NASA/CALCE/Oxford caches) typically expose:

1. "summary" view — one row per cycle with hand-crafted scalar features
   (classic ML: RF/SVR/etc, and also usable as the per-timestep input to an
   LSTM operating over cycles).
2. "sequence" view — the raw per-point charge/discharge curves for each
   cycle, resampled to a fixed length, for models that want the full
   voltage/current/temperature curve shape rather than summary stats
   (closer to what CALCE/Oxford curve-based LSTM approaches use).

SOH definition
--------------
Coin cells often show a rising capacity over the first several cycles
(formation/break-in) before entering monotonic fade. Using cycle 1 as the
SOH reference (as is conventional for 18650-type datasets like NASA/CALCE)
would be misleading here since cycle 1 is not the cell's peak capacity.

We therefore define:
    C_ref = max discharge capacity observed in the first `formation_window`
            cycles (default 10), i.e. the practical "beginning of life"
            capacity after break-in.
    SOH(cycle) = discharge_capacity(cycle) / C_ref * 100

This is recorded alongside the raw discharge capacity and the alternative
cycle-1-referenced SOH, so downstream consumers can pick either convention.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

DEFAULT_FORMATION_WINDOW = 10
DEFAULT_EOL_THRESHOLD = 80.0  # % SOH, conventional knee-point definition
DEFAULT_SEQ_LEN = 128  # resample length for per-cycle curve sequences


def discharge_capacity_per_cycle(df: pd.DataFrame) -> pd.Series:
    """Max Capacity(Ah) reached during the CC DChg step of each cycle.

    Capacity resets to 0 at the start of each step in this Neware export
    convention (confirmed empirically: SOC/DOD reaches exactly 100% at the
    end of each discharge), so the max value within the CC DChg rows of a
    cycle is that cycle's discharged capacity.
    """
    dchg = df[df["step_type"] == "CC DChg"]
    return dchg.groupby("cycle")["capacity_ah"].max()


def charge_capacity_per_cycle(df: pd.DataFrame) -> pd.Series:
    """Max Capacity(Ah) reached during CC Chg + CV Chg combined (total charge in)."""
    chg = df[df["step_type"].isin(["CC Chg", "CV Chg"])]
    # capacity resets between CC Chg and CV Chg steps too, so sum each step's max
    per_step_max = chg.groupby(["cycle", "step_index"])["capacity_ah"].max()
    return per_step_max.groupby("cycle").sum()


@dataclass
class SOHResult:
    table: pd.DataFrame       # per-cycle summary + SOH labels
    c_ref: float               # reference (beginning-of-life) capacity, Ah
    c_rated_guess: float | None  # optional externally-supplied nominal capacity


def compute_soh_table(
    df: pd.DataFrame,
    formation_window: int = DEFAULT_FORMATION_WINDOW,
    c_rated: float | None = None,
) -> SOHResult:
    dchg_cap = discharge_capacity_per_cycle(df)
    chg_cap = charge_capacity_per_cycle(df)
    cycles = sorted(df["cycle"].unique())

    formation_cycles = [c for c in cycles if c <= formation_window]
    c_ref = float(dchg_cap.loc[formation_cycles].max())
    c_cycle1 = float(dchg_cap.loc[cycles[0]])

    coulombic_eff = (dchg_cap / chg_cap).clip(upper=1.5)  # guard against div-by-~0 spikes

    table = pd.DataFrame({
        "cycle": cycles,
        "discharge_capacity_ah": dchg_cap.reindex(cycles).values,
        "charge_capacity_ah": chg_cap.reindex(cycles).values,
        "coulombic_efficiency": coulombic_eff.reindex(cycles).values,
        "soh_pct": (dchg_cap.reindex(cycles) / c_ref * 100).values,
        "soh_pct_vs_cycle1": (dchg_cap.reindex(cycles) / c_cycle1 * 100).values,
    })
    if c_rated:
        table["soh_pct_vs_rated"] = dchg_cap.reindex(cycles).values / c_rated * 100

    return SOHResult(table=table, c_ref=c_ref, c_rated_guess=c_rated)


def find_eol_cycle(soh_table: pd.DataFrame, threshold: float = DEFAULT_EOL_THRESHOLD) -> int | None:
    """First cycle at/after formation where SOH drops to `threshold`% and stays
    at or below it for the remainder (avoids flagging noisy single-cycle dips).
    Returns None if the threshold is never sustained.
    """
    below = soh_table["soh_pct"].values <= threshold
    for i in range(len(below)):
        if below[i:].all():
            return int(soh_table["cycle"].iloc[i])
    return None


def _resample_curve(x: np.ndarray, y: np.ndarray, n: int) -> np.ndarray:
    """Resample y(x) onto n evenly spaced points spanning x's own range.
    x must be monotonically non-decreasing (elapsed time within a step).
    """
    if len(x) < 2:
        return np.full(n, y[0] if len(y) else np.nan)
    x_new = np.linspace(x[0], x[-1], n)
    return np.interp(x_new, x, y)


def extract_cycle_sequences(
    df: pd.DataFrame,
    seq_len: int = DEFAULT_SEQ_LEN,
) -> dict[str, np.ndarray]:
    """Build fixed-length resampled curves per cycle for the discharge and
    charge (CC+CV) phases: voltage, current, temperature vs. normalized time.

    Returns a dict of arrays each shaped (n_cycles, seq_len):
        dchg_voltage, dchg_current, dchg_temperature,
        chg_voltage, chg_current, chg_temperature
    plus 'cycles' (n_cycles,) giving the cycle index each row corresponds to.
    """
    cycles = sorted(df["cycle"].unique())
    out = {
        k: np.full((len(cycles), seq_len), np.nan)
        for k in [
            "dchg_voltage", "dchg_current", "dchg_temperature",
            "chg_voltage", "chg_current", "chg_temperature",
        ]
    }

    for i, c in enumerate(cycles):
        cyc_df = df[df["cycle"] == c]

        dchg = cyc_df[cyc_df["step_type"] == "CC DChg"]
        if len(dchg) >= 2:
            t = dchg["elapsed_s"].to_numpy(dtype=float)
            t = t - t[0]
            out["dchg_voltage"][i] = _resample_curve(t, dchg["voltage_v"].to_numpy(dtype=float), seq_len)
            out["dchg_current"][i] = _resample_curve(t, dchg["current_a"].to_numpy(dtype=float), seq_len)
            out["dchg_temperature"][i] = _resample_curve(t, dchg["temperature_c"].to_numpy(dtype=float), seq_len)

        chg = cyc_df[cyc_df["step_type"].isin(["CC Chg", "CV Chg"])]
        if len(chg) >= 2:
            t = chg["elapsed_s"].to_numpy(dtype=float)
            t = t - t[0]
            out["chg_voltage"][i] = _resample_curve(t, chg["voltage_v"].to_numpy(dtype=float), seq_len)
            out["chg_current"][i] = _resample_curve(t, chg["current_a"].to_numpy(dtype=float), seq_len)
            out["chg_temperature"][i] = _resample_curve(t, chg["temperature_c"].to_numpy(dtype=float), seq_len)

    out["cycles"] = np.array(cycles, dtype=np.int32)
    return out


def extract_cycle_scalar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Additional per-cycle scalar features useful for the summary/ML view,
    beyond raw/relative capacity: step durations, temperature stats,
    IC-curve-adjacent stats, and end-of-charge/discharge voltages.
    """
    rows = []
    for c, cyc_df in df.groupby("cycle"):
        dchg = cyc_df[cyc_df["step_type"] == "CC DChg"]
        cc_chg = cyc_df[cyc_df["step_type"] == "CC Chg"]
        cv_chg = cyc_df[cyc_df["step_type"] == "CV Chg"]
        rest = cyc_df[cyc_df["step_type"] == "Rest"]

        def duration_s(sub):
            return float(sub["elapsed_s"].max() - sub["elapsed_s"].min()) if len(sub) >= 2 else 0.0

        def mean_temp(sub):
            return float(sub["temperature_c"].mean()) if len(sub) else np.nan

        rows.append({
            "cycle": c,
            "dchg_duration_s": duration_s(dchg),
            "cc_chg_duration_s": duration_s(cc_chg),
            "cv_chg_duration_s": duration_s(cv_chg),
            "mean_temperature_c": mean_temp(cyc_df[cyc_df["step_type"] != "Rest"]),
            "max_temperature_c": float(cyc_df["temperature_c"].max()),
            "min_dchg_voltage_v": float(dchg["voltage_v"].min()) if len(dchg) else np.nan,
            "max_chg_voltage_v": float(cc_chg["voltage_v"].max()) if len(cc_chg) else np.nan,
            "cv_chg_end_current_a": float(cv_chg["current_a"].iloc[-1]) if len(cv_chg) else np.nan,
            "mean_rest_voltage_v": float(rest["voltage_v"].mean()) if len(rest) else np.nan,
        })
    return pd.DataFrame(rows).sort_values("cycle").reset_index(drop=True)
