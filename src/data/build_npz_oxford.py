"""
Adapter: convert an OXFORD_*_raw.npz cache (raw per-sample V/I/T time series,
phase-coded, one file per cell) into the same standardized per-cell .npz
schema used for the VL coin cells (see build_npz.py's module docstring),
so both datasets can be pooled through the same dataset.py / train.py /
export_web_json.py pipeline unchanged.

Why an adapter instead of touching the VL schema
--------------------------------------------------
The VL npz schema was designed around Neware's step-typed export (explicit
CC Chg / CV Chg / CC DChg / Rest step labels). The Oxford raw cache instead
gives a continuous phase_code per sample (+1 charge, -1 discharge, no rest
phase at all, no explicit CV flag): confirmed empirically across all 8 cells
here (np.unique(phase_code) == [-1, 1] everywhere). Oxford's charge phase is
a CCCV profile without a labeled CV segment: current holds at ~rated C-rate
for most of the charge, then tapers as voltage approaches the 4.2V charge
limit. We detect that taper empirically (see `_split_cccv_tail`) to recover
a CV-equivalent duration and end-current, so those two VL features keep a
meaningful (if approximate) counterpart. There is no rest phase in this
data, so `mean_rest_voltage_v` cannot be computed and is filled with NaN
(train.py callers must treat NaN static/summary columns accordingly, or the
column is simply dropped for pooled training — see README note in train.py
change).

SOH definition — kept dataset-native, NOT re-derived
-----------------------------------------------------
Unlike the VL cells (which need a custom post-formation C_ref because they
show a break-in capacity rise), the Oxford cache already ships a per-cycle
`SOH` array (capacity_Ah / rated, using Oxford's own nominal 0.74 Ah rating)
that is the dataset's own established convention in the literature. We keep
it as-is rather than recomputing a VL-style "peak-of-first-10-cycles"
reference, since Oxford cells do not show the same break-in rise and
inventing a different reference would just diverge from how these cells are
normally reported. soh_pct = SOH * 100.

Usage:
    python build_npz_oxford.py <input_raw.npz> [<input2_raw.npz> ...] --out-dir ../../data/processed
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

DEFAULT_SEQ_LEN = 128


def _resample_curve(x: np.ndarray, y: np.ndarray, n: int) -> np.ndarray:
    if len(x) < 2:
        return np.full(n, y[0] if len(y) else np.nan)
    x_new = np.linspace(x[0], x[-1], n)
    return np.interp(x_new, x, y)


def _split_cccv_tail(t: np.ndarray, i: np.ndarray, rated: float, frac_threshold: float = 0.98):
    """Given a charge phase's time/current arrays, find where current first
    sustainably drops below `frac_threshold` * its own early-phase plateau
    value (the CC->CV transition), returning (cv_start_index, cv_duration_s,
    cv_end_current_a). If no such drop is found (fully CC to the end, as
    happens on some early/high-SOH cycles), returns (len(i), 0.0, i[-1]).
    """
    if len(i) < 5:
        return len(i), 0.0, float(i[-1]) if len(i) else np.nan
    plateau = np.median(i[: max(3, len(i) // 10)])
    thresh = plateau * frac_threshold
    below = i < thresh
    # require the drop to persist (avoid single-sample noise): first index
    # after which `below` stays true for the rest of the phase
    idx = None
    for k in range(len(below)):
        if below[k:].all():
            idx = k
            break
    if idx is None:
        return len(i), 0.0, float(i[-1])
    cv_dur = float(t[-1] - t[idx])
    return idx, cv_dur, float(i[-1])


def build_npz_for_oxford_raw(
    raw_path: str | Path,
    out_dir: str | Path,
    seq_len: int = DEFAULT_SEQ_LEN,
    eol_threshold_pct: float = 80.0,
) -> Path:
    raw_path = Path(raw_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    d = np.load(raw_path, allow_pickle=True)
    name = str(d["name"])
    cell_id = f"OXFORD_{name}"
    rated = float(d["rated_capacity_Ah"])
    n_cycles = int(d["n_cycles"])
    cycle_number = d["cycle_number"]
    capacity_ah = d["capacity_Ah"]
    soh = d["SOH"]  # already capacity_Ah / rated, dataset-native convention

    cycle_ptr = d["cycle_ptr"]
    time_s = d["time_s"]
    voltage_v = d["voltage_V"]
    current_a = d["current_A"]
    temperature_c = d["temperature_C"]
    phase_code = d["phase_code"]

    cycles_out = np.arange(1, n_cycles + 1, dtype=np.int32)  # re-index 1..N for display

    soh_pct = (soh * 100.0).astype(np.float64)

    dchg_duration_s = np.full(n_cycles, np.nan)
    cc_chg_duration_s = np.full(n_cycles, np.nan)
    cv_chg_duration_s = np.full(n_cycles, np.nan)
    mean_temperature_c = np.full(n_cycles, np.nan)
    max_temperature_c = np.full(n_cycles, np.nan)
    min_dchg_voltage_v = np.full(n_cycles, np.nan)
    max_chg_voltage_v = np.full(n_cycles, np.nan)
    cv_chg_end_current_a = np.full(n_cycles, np.nan)
    mean_rest_voltage_v = np.full(n_cycles, np.nan)  # no rest phase in this dataset
    # coulombic efficiency: Oxford raw has no separate charge-Ah counter, so
    # approximate via elapsed-time*current integration per phase.
    coulombic_efficiency = np.full(n_cycles, np.nan)

    seqs = {k: np.full((n_cycles, seq_len), np.nan) for k in [
        "dchg_voltage", "dchg_current", "dchg_temperature",
        "chg_voltage", "chg_current", "chg_temperature",
    ]}

    for idx in range(n_cycles):
        s, e = cycle_ptr[idx], cycle_ptr[idx + 1]
        t = time_s[s:e]
        v = voltage_v[s:e]
        i = current_a[s:e]
        temp = temperature_c[s:e]
        ph = phase_code[s:e]

        mean_temperature_c[idx] = float(np.nanmean(temp)) if len(temp) else np.nan
        max_temperature_c[idx] = float(np.nanmax(temp)) if len(temp) else np.nan

        chg_mask = ph == 1
        dis_mask = ph == -1

        if chg_mask.sum() >= 2:
            ct = t[chg_mask]; ct = ct - ct[0]
            cv = v[chg_mask]; ci = i[chg_mask]; ctemp = temp[chg_mask]
            cv_start_idx, cv_dur, cv_end_i = _split_cccv_tail(ct, ci, rated)
            cc_chg_duration_s[idx] = float(ct[cv_start_idx - 1]) if cv_start_idx > 0 else 0.0
            cv_chg_duration_s[idx] = cv_dur
            cv_chg_end_current_a[idx] = cv_end_i
            max_chg_voltage_v[idx] = float(cv.max())
            seqs["chg_voltage"][idx] = _resample_curve(ct, cv, seq_len)
            seqs["chg_current"][idx] = _resample_curve(ct, ci, seq_len)
            seqs["chg_temperature"][idx] = _resample_curve(ct, ctemp, seq_len)
            charge_ah = float(np.trapezoid(ci, ct) / 3600.0)  # A * s -> Ah
        else:
            charge_ah = np.nan

        if dis_mask.sum() >= 2:
            dt = t[dis_mask]; dt = dt - dt[0]
            dv = v[dis_mask]; di = i[dis_mask]; dtemp = temp[dis_mask]
            dchg_duration_s[idx] = float(dt[-1] - dt[0])
            min_dchg_voltage_v[idx] = float(dv.min())
            seqs["dchg_voltage"][idx] = _resample_curve(dt, dv, seq_len)
            seqs["dchg_current"][idx] = _resample_curve(dt, di, seq_len)
            seqs["dchg_temperature"][idx] = _resample_curve(dt, dtemp, seq_len)
            discharge_ah = float(abs(np.trapezoid(di, dt)) / 3600.0)
        else:
            discharge_ah = np.nan

        if charge_ah and charge_ah > 1e-6 and not np.isnan(discharge_ah):
            coulombic_efficiency[idx] = min(discharge_ah / charge_ah, 1.5)

    # EOL: first sustained cycle at/below threshold
    below = soh_pct <= eol_threshold_pct
    eol = -1
    for k in range(len(below)):
        if below[k:].all():
            eol = int(cycles_out[k])
            break

    payload = dict(
        cell_id=cell_id,
        form_factor="OXFORD",
        c_rate="0.5C",   # Oxford Ch1: ~0.74A CC on a 0.74Ah nominal cell ≈ ~1C rate;
                          # documented cell is cycled at C/2 CC-CV per the Oxford
                          # battery degradation dataset protocol — kept as a fixed
                          # static tag consistent across all 8 Oxford cells here.
        temp_condition="40",  # Oxford dataset cycling temperature (40°C ambient)
        c_ref_ah=np.float64(rated),
        formation_window=np.int64(0),  # not applicable; kept for schema parity
        eol_cycle_80pct=np.int64(eol),
        cycles=cycles_out,
        discharge_capacity_ah=capacity_ah.astype(np.float64),
        charge_capacity_ah=capacity_ah.astype(np.float64),  # not independently measured; placeholder equal to discharge
        coulombic_efficiency=coulombic_efficiency.astype(np.float64),
        soh_pct=soh_pct,
        soh_pct_vs_cycle1=(capacity_ah / capacity_ah[0] * 100.0).astype(np.float64),
        dchg_duration_s=dchg_duration_s,
        cc_chg_duration_s=cc_chg_duration_s,
        cv_chg_duration_s=cv_chg_duration_s,
        mean_temperature_c=mean_temperature_c,
        max_temperature_c=max_temperature_c,
        min_dchg_voltage_v=min_dchg_voltage_v,
        max_chg_voltage_v=max_chg_voltage_v,
        cv_chg_end_current_a=cv_chg_end_current_a,
        mean_rest_voltage_v=mean_rest_voltage_v,
        dchg_voltage=seqs["dchg_voltage"].astype(np.float32),
        dchg_current=seqs["dchg_current"].astype(np.float32),
        dchg_temperature=seqs["dchg_temperature"].astype(np.float32),
        chg_voltage=seqs["chg_voltage"].astype(np.float32),
        chg_current=seqs["chg_current"].astype(np.float32),
        chg_temperature=seqs["chg_temperature"].astype(np.float32),
        seq_len=np.int64(seq_len),
        source="oxford_raw_adapter",
    )

    out_path = out_dir / f"{cell_id}.npz"
    np.savez_compressed(out_path, **payload)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--out-dir", default="../../data/processed")
    args = ap.parse_args()
    for inp in args.inputs:
        try:
            out_path = build_npz_for_oxford_raw(inp, args.out_dir)
            print(f"OK  {inp} -> {out_path}")
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"FAIL {inp}: {e}")


if __name__ == "__main__":
    main()
