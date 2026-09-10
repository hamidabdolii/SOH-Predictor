"""
Convert a Neware VL coin-cell xlsx export into a standardized .npz cache.

Usage:
    python build_npz.py <input.xlsx> [<input2.xlsx> ...] --out-dir data/processed

Output schema (per cell, one .npz file per input xlsx)
-------------------------------------------------------
Scalars / metadata:
    cell_id              str            e.g. "VL2020-0.2C-RT"
    form_factor          str            e.g. "VL2020"
    c_rate               str            e.g. "0.2C"
    temp_condition       str            "RT" or nominal degC, e.g. "40"
    c_ref_ah             float64        reference (post-formation) capacity, Ah
    formation_window     int64          cycles used to determine c_ref
    eol_cycle_80pct       int64 or -1    first sustained-below-80%-SOH cycle (-1 if none)

Per-cycle summary arrays, all shape (n_cycles,):
    cycles                       int32
    discharge_capacity_ah        float64
    charge_capacity_ah           float64
    coulombic_efficiency         float64
    soh_pct                       float64   <-- primary SOH label (vs. c_ref)
    soh_pct_vs_cycle1            float64
    dchg_duration_s               float64
    cc_chg_duration_s             float64
    cv_chg_duration_s             float64
    mean_temperature_c            float64
    max_temperature_c             float64
    min_dchg_voltage_v            float64
    max_chg_voltage_v             float64
    cv_chg_end_current_a          float64
    mean_rest_voltage_v           float64

Per-cycle resampled curves, all shape (n_cycles, seq_len):
    dchg_voltage, dchg_current, dchg_temperature
    chg_voltage, chg_current, chg_temperature

This layout intentionally keeps every field as a flat named array (no nested
dicts/objects) so it loads identically to typical NASA/CALCE/Oxford npz
caches via `np.load(path)` and each field is directly usable as a numpy
array without pickling. If the actual NASA/CALCE/Oxford caches use different
key names, add a thin renaming/adapter layer at data-loading time in the app
rather than changing this schema — keeping the VL schema self-consistent and
documented is more valuable than guessing at an unseen convention.
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np

from parse_neware import read_neware_xlsx, infer_meta, validate_cycle_structure
from soh_features import (
    compute_soh_table,
    find_eol_cycle,
    extract_cycle_sequences,
    extract_cycle_scalar_features,
    DEFAULT_FORMATION_WINDOW,
    DEFAULT_SEQ_LEN,
    DEFAULT_EOL_THRESHOLD,
)


def build_npz_for_file(
    xlsx_path: str | Path,
    out_dir: str | Path,
    formation_window: int = DEFAULT_FORMATION_WINDOW,
    seq_len: int = DEFAULT_SEQ_LEN,
    eol_threshold: float = DEFAULT_EOL_THRESHOLD,
    c_rated: float | None = None,
) -> Path:
    xlsx_path = Path(xlsx_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = read_neware_xlsx(xlsx_path)
    meta = infer_meta(df)
    diag = validate_cycle_structure(df)

    if diag["missing_cycles"]:
        raise ValueError(f"{xlsx_path.name}: missing cycles {diag['missing_cycles']}, refusing to build npz")

    # Cycle 1 is excluded from all downstream analysis: it carries an extra
    # leading "Rest" step not present in any other cycle (an artifact of the
    # test setup, not the cycling protocol), and its discharge capacity is
    # visibly inconsistent with the rest of the curve on at least one cell
    # (e.g. VL2020-0.2C-RT: cycle 1 = 0.0016 Ah vs. cycle 2 = 0.0148 Ah, a
    # ~9x jump that cycle 2 onward doesn't repeat). Analysis and the SOH
    # reference/EOL calculations below all start from cycle 2.
    df = df[df["cycle"] != 1].reset_index(drop=True)

    soh_res = compute_soh_table(df, formation_window=formation_window, c_rated=c_rated)
    eol = find_eol_cycle(soh_res.table, threshold=eol_threshold)
    seqs = extract_cycle_sequences(df, seq_len=seq_len)
    scalars = extract_cycle_scalar_features(df)

    # sanity: cycle alignment across all per-cycle sources
    assert list(soh_res.table["cycle"]) == list(scalars["cycle"]), "cycle misalignment: soh vs scalar features"
    assert list(soh_res.table["cycle"]) == list(seqs["cycles"]), "cycle misalignment: soh vs sequences"

    payload = dict(
        cell_id=meta.cell_id,
        form_factor=meta.form_factor,
        c_rate=meta.c_rate,
        temp_condition=meta.temp_condition,
        c_ref_ah=np.float64(soh_res.c_ref),
        formation_window=np.int64(formation_window),
        eol_cycle_80pct=np.int64(eol if eol is not None else -1),
        cycles=soh_res.table["cycle"].to_numpy(dtype=np.int32),
        discharge_capacity_ah=soh_res.table["discharge_capacity_ah"].to_numpy(dtype=np.float64),
        charge_capacity_ah=soh_res.table["charge_capacity_ah"].to_numpy(dtype=np.float64),
        coulombic_efficiency=soh_res.table["coulombic_efficiency"].to_numpy(dtype=np.float64),
        soh_pct=soh_res.table["soh_pct"].to_numpy(dtype=np.float64),
        soh_pct_vs_cycle1=soh_res.table["soh_pct_vs_cycle1"].to_numpy(dtype=np.float64),
        dchg_duration_s=scalars["dchg_duration_s"].to_numpy(dtype=np.float64),
        cc_chg_duration_s=scalars["cc_chg_duration_s"].to_numpy(dtype=np.float64),
        cv_chg_duration_s=scalars["cv_chg_duration_s"].to_numpy(dtype=np.float64),
        mean_temperature_c=scalars["mean_temperature_c"].to_numpy(dtype=np.float64),
        max_temperature_c=scalars["max_temperature_c"].to_numpy(dtype=np.float64),
        min_dchg_voltage_v=scalars["min_dchg_voltage_v"].to_numpy(dtype=np.float64),
        max_chg_voltage_v=scalars["max_chg_voltage_v"].to_numpy(dtype=np.float64),
        cv_chg_end_current_a=scalars["cv_chg_end_current_a"].to_numpy(dtype=np.float64),
        mean_rest_voltage_v=scalars["mean_rest_voltage_v"].to_numpy(dtype=np.float64),
        dchg_voltage=seqs["dchg_voltage"].astype(np.float32),
        dchg_current=seqs["dchg_current"].astype(np.float32),
        dchg_temperature=seqs["dchg_temperature"].astype(np.float32),
        chg_voltage=seqs["chg_voltage"].astype(np.float32),
        chg_current=seqs["chg_current"].astype(np.float32),
        chg_temperature=seqs["chg_temperature"].astype(np.float32),
        seq_len=np.int64(seq_len),
    )

    out_path = out_dir / f"{meta.cell_id}.npz"
    np.savez_compressed(out_path, **payload)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="input xlsx file(s)")
    ap.add_argument("--out-dir", default="data/processed")
    ap.add_argument("--formation-window", type=int, default=DEFAULT_FORMATION_WINDOW)
    ap.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN)
    ap.add_argument("--eol-threshold", type=float, default=DEFAULT_EOL_THRESHOLD)
    args = ap.parse_args()

    for inp in args.inputs:
        try:
            out_path = build_npz_for_file(
                inp,
                args.out_dir,
                formation_window=args.formation_window,
                seq_len=args.seq_len,
                eol_threshold=args.eol_threshold,
            )
            print(f"OK  {inp} -> {out_path}")
        except Exception as e:
            print(f"FAIL {inp}: {e}")


if __name__ == "__main__":
    main()
