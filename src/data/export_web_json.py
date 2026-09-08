"""
Convert processed VL npz caches into the compact per-cell JSON assets the
static web app loads directly (no server, no build step). Also writes
catalog.json, the index the cell-selection UI reads.

Also folds in the leave-one-cell-out (LOCO) predictions produced by
train.py (models/loco_predictions.json) as each cell's `loco` field — these
are genuine out-of-sample predictions (that cell's data was held out of
training entirely for its own fold) and are what the web app's "predicted
vs actual" trajectory chart displays. Do NOT substitute in-browser
predictions from the single production model here: that model was trained
on every cell including this one, so its predictions on it would not be a
fair out-of-sample test.

Usage:
    python export_web_json.py --npz-dir ../../data/processed --out-dir ../../web/data --loco-predictions ../../models/loco_predictions.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def export_all(npz_dir: str, out_dir: str, loco_predictions_path: str | None = None):
    npz_dir = Path(npz_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    loco_predictions = {}
    if loco_predictions_path and Path(loco_predictions_path).exists():
        with open(loco_predictions_path) as f:
            loco_predictions = json.load(f)
    else:
        print(f"WARNING: no LOCO predictions file found at {loco_predictions_path} — "
              f"cells will ship without a 'loco' field, and the trajectory chart will "
              f"have nothing to show for them.")

    catalog = []
    for npz_path in sorted(npz_dir.glob("*.npz")):
        d = np.load(npz_path)
        cell_id = str(d["cell_id"])
        n = len(d["cycles"])

        cell_data = {
            "cell_id": cell_id,
            "form_factor": str(d["form_factor"]),
            "c_rate": str(d["c_rate"]),
            "temp_condition": str(d["temp_condition"]),
            "c_ref_ah": float(d["c_ref_ah"]),
            "eol_cycle_80pct": int(d["eol_cycle_80pct"]),
            "cycles": d["cycles"].tolist(),
            "soh_pct": [round(float(x), 3) for x in d["soh_pct"]],
            "discharge_capacity_ah": [round(float(x), 8) for x in d["discharge_capacity_ah"]],
            "discharge_capacity_mah": [round(float(x) * 1000, 5) for x in d["discharge_capacity_ah"]],
            "charge_capacity_mah": [round(float(x) * 1000, 5) for x in d["charge_capacity_ah"]],
            "coulombic_efficiency": [round(float(x), 4) for x in d["coulombic_efficiency"]],
            "mean_temperature_c": [round(float(x), 2) for x in d["mean_temperature_c"]],
            "dchg_duration_s": [round(float(x), 1) for x in d["dchg_duration_s"]],
            "cc_chg_duration_s": [round(float(x), 1) for x in d["cc_chg_duration_s"]],
            "cv_chg_duration_s": [round(float(x), 1) for x in d["cv_chg_duration_s"]],
            "max_temperature_c": [round(float(x), 2) for x in d["max_temperature_c"]],
            "min_dchg_voltage_v": [round(float(x), 4) for x in d["min_dchg_voltage_v"]],
            "max_chg_voltage_v": [round(float(x), 4) for x in d["max_chg_voltage_v"]],
            "cv_chg_end_current_a": [round(float(x), 6) for x in d["cv_chg_end_current_a"]],
            # Full per-cycle curves for EVERY cycle (not just a handful of
            # samples): voltage, current, and temperature for both the
            # discharge and charge phases, resampled to a fixed length.
            # ~99 cycles x 128 points x 6 series is small enough (roughly
            # half a megabyte per cell as JSON) to ship in full and let the
            # UI show any cycle the user picks, not just pre-selected ones.
            "curves": {},
            # Out-of-sample leave-one-cell-out predictions for this cell, if
            # available (see module docstring). null when not available —
            # the frontend should show "not available" rather than fabricate
            # a chart in that case.
            "loco": loco_predictions.get(cell_id),
        }

        for idx in range(n):
            cyc = int(d["cycles"][idx])
            cell_data["curves"][str(cyc)] = {
                "dchg_voltage": [round(float(x), 4) for x in d["dchg_voltage"][idx]],
                "dchg_current": [round(float(x), 6) for x in d["dchg_current"][idx]],
                "dchg_temperature": [round(float(x), 3) for x in d["dchg_temperature"][idx]],
                "chg_voltage": [round(float(x), 4) for x in d["chg_voltage"][idx]],
                "chg_current": [round(float(x), 6) for x in d["chg_current"][idx]],
                "chg_temperature": [round(float(x), 3) for x in d["chg_temperature"][idx]],
            }

        with open(out_dir / f"{cell_id}.json", "w") as f:
            json.dump(cell_data, f)

        catalog.append({
            "cell_id": cell_id,
            "form_factor": cell_data["form_factor"],
            "c_rate": cell_data["c_rate"],
            "temp_condition": cell_data["temp_condition"],
            "n_cycles": n,
            "eol_cycle_80pct": cell_data["eol_cycle_80pct"],
        })

    with open(out_dir / "catalog.json", "w") as f:
        json.dump(catalog, f, indent=2)

    print(f"Wrote {len(catalog)} cell JSON files + catalog.json to {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz-dir", default="../../data/processed")
    ap.add_argument("--out-dir", default="../../web/data")
    ap.add_argument("--loco-predictions", default="../../models/loco_predictions.json")
    args = ap.parse_args()
    export_all(args.npz_dir, args.out_dir, args.loco_predictions)
