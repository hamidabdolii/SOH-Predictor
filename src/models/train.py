"""
Train the SOH-LSTM with leave-one-cell-out (LOCO) cross-validation across
the available VL npz caches, and compare against a Random Forest baseline
on the same windowed features (flattened) to sanity-check that the LSTM is
actually earning its complexity.

With only 4 (eventually 6) cells, LOCO is the only defensible way to see
how well this generalizes to a *battery the model has never trained on* —
a random train/test split on windows would leak information between
overlapping windows of the same cell and give an optimistic, meaningless
score.

Usage:
    python train.py --npz-dir ../../data/processed --out-dir ../../models
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from dataset import (
    load_cell_record,
    make_windows,
    Normalizer,
    StaticNormalizer,
    SEQUENCE_FEATURE_KEYS,
)
from lstm_soh import SOHRecurrent, LSTMConfig, ARCH_CHOICES, ARCH_LABELS

torch.manual_seed(0)
np.random.seed(0)


SOH_IDX = SEQUENCE_FEATURE_KEYS.index("soh_pct")


def samples_to_tensors(samples, seq_norm, static_norm):
    x_seq_raw = np.stack([s.x_seq for s in samples]).astype(np.float32)
    x_seq = np.stack([seq_norm.transform(s.x_seq) for s in samples]).astype(np.float32)
    x_static = np.stack([static_norm.transform(s.x_static) for s in samples]).astype(np.float32)
    y = np.array([s.y for s in samples], dtype=np.float32)
    last_soh = x_seq_raw[:, -1, SOH_IDX]
    return (
        torch.from_numpy(x_seq),
        torch.from_numpy(x_static),
        torch.from_numpy(y),
        torch.from_numpy(last_soh),
    )


def train_one_fold(
    train_records,
    test_samples,
    n_features,
    n_static,
    epochs=400,
    lr=1e-3,
    patience=30,
    hidden_size=16,
    dropout=0.3,
    weight_decay=1e-4,
    arch="lstm",
    window=10,
    horizon=1,
):
    """Train on `train_records` (cells), evaluate on `test_samples` (the
    held-out cell's windows). Early stopping uses a validation split made by
    holding out ONE of the training cells in turn (never the true test
    cell), so model selection never sees the test cell's data.
    """
    if len(train_records) >= 2:
        # use the last training cell as the early-stopping validation cell
        es_val_record = train_records[-1]
        fit_records = train_records[:-1]
        es_val_samples = make_windows(es_val_record, window=window, horizon=horizon)
    else:
        fit_records = train_records
        es_val_samples = make_windows(train_records[0], window=window, horizon=horizon)

    fit_samples = []
    for r in fit_records:
        fit_samples += make_windows(r, window=window, horizon=horizon)

    seq_norm = Normalizer.fit(fit_samples)
    static_norm = StaticNormalizer.fit(fit_samples)

    x_seq_tr, x_static_tr, y_tr, last_soh_tr = samples_to_tensors(fit_samples, seq_norm, static_norm)
    x_seq_es, x_static_es, y_es, last_soh_es = samples_to_tensors(es_val_samples, seq_norm, static_norm)
    x_seq_te, x_static_te, y_te, last_soh_te = samples_to_tensors(test_samples, seq_norm, static_norm)

    cfg = LSTMConfig(
        n_features=n_features,
        n_static=n_static,
        soh_feature_idx=SOH_IDX,
        hidden_size=hidden_size,
        dropout=dropout,
        arch=arch,
    )
    model = SOHRecurrent(cfg)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.SmoothL1Loss()  # Huber: more robust to the occasional noisy cycle

    best_val = float("inf")
    best_state = None
    epochs_no_improve = 0

    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        pred = model(x_seq_tr, x_static_tr, last_soh_tr)
        loss = loss_fn(pred, y_tr)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            es_pred = model(x_seq_es, x_static_es, last_soh_es)
            es_loss = loss_fn(es_pred, y_es).item()

        if es_loss < best_val - 1e-4:
            best_val = es_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        final_pred = model(x_seq_te, x_static_te, last_soh_te).numpy()

    return model, seq_norm, static_norm, final_pred, y_te.numpy()


def naive_persistence(val_samples):
    """Predict SOH(target) = SOH at the last cycle of the input window.
    The floor any model must beat to be worth using."""
    pred = np.array([s.x_seq[-1, SOH_IDX] for s in val_samples])
    y = np.array([s.y for s in val_samples])
    return pred, y


def rf_baseline(train_samples, val_samples):
    def flatten(samples):
        X = np.stack([s.x_seq.flatten() for s in samples])
        static = np.stack([s.x_static for s in samples])
        X = np.concatenate([X, static], axis=1)
        y = np.array([s.y for s in samples])
        return X, y

    X_tr, y_tr = flatten(train_samples)
    X_va, y_va = flatten(val_samples)
    rf = RandomForestRegressor(n_estimators=300, max_depth=6, min_samples_leaf=3, random_state=0)
    rf.fit(X_tr, y_tr)
    pred = rf.predict(X_va)
    return pred, y_va


def metrics(y_true, y_pred) -> dict:
    """MAE / RMSE / max error / R2 for one held-out cell.

    A note on R2 here: it is computed against the variance of *that cell's
    own* target SOH values. That makes it a fair "does the model explain
    this cell's trajectory" score, but it is NOT comparable across cells
    with very different fade spans — a cell whose SOH barely moves has tiny
    denominator variance, so even small absolute errors can drag R2 down
    (or negative). `soh_span` is reported alongside precisely so that a low
    R2 on a flat cell can be read in context rather than mistaken for a
    worse fit than a high-R2 fast-fading cell. MAE/RMSE, being absolute,
    stay comparable across cells; R2 is the relative complement to them.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    span = float(np.max(y_true) - np.min(y_true))
    # r2_score is undefined (0/0) when every target is identical; guard it.
    r2 = float(r2_score(y_true, y_pred)) if span > 1e-9 else float("nan")
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "max_err": float(np.max(np.abs(y_true - y_pred))),
        "r2": r2,
        "soh_span": span,
        "n": int(len(y_true)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz-dir", default="../../data/processed")
    ap.add_argument("--out-dir", default="../../models")
    ap.add_argument("--window", type=int, default=10)
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument(
        "--archs",
        default=",".join(ARCH_CHOICES),
        help=f"comma-separated recurrent backbones to evaluate (any of {ARCH_CHOICES})",
    )
    args = ap.parse_args()

    archs = [a.strip() for a in args.archs.split(",") if a.strip()]
    for a in archs:
        if a not in ARCH_CHOICES:
            raise SystemExit(f"Unknown arch {a!r}; expected any of {ARCH_CHOICES}")

    npz_dir = Path(args.npz_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    npz_files = sorted(npz_dir.glob("*.npz"))
    npz_files = [f for f in npz_files if f.stem not in ("soh_curves_check",)]
    records = [load_cell_record(str(f)) for f in npz_files]
    print(f"Loaded {len(records)} cells: {[r.cell_id for r in records]}")

    n_features = len(SEQUENCE_FEATURE_KEYS)
    n_static = 3

    # method key -> {cell_id -> metrics}. Recurrent architectures each get
    # their own key ("lstm", "gru", ...) alongside the two non-neural
    # baselines, so every method in the table was scored under the identical
    # leave-one-cell-out protocol.
    loco_results = {a: {} for a in archs}
    loco_results["rf"] = {}
    loco_results["naive"] = {}

    # arch -> cell_id -> {target_cycle, true_soh_pct, predicted_soh_pct}.
    # These are genuine out-of-sample predictions (that cell's data was held
    # out of training entirely for its own fold), safe to ship to the web
    # app's "predicted vs actual" trajectory chart. The production models
    # trained on ALL cells (below) must NOT be used for that display: they
    # have seen every cell, so their predictions are not a fair test.
    loco_predictions = {a: {} for a in archs}

    for held_out in records:
        train_records = [r for r in records if r.cell_id != held_out.cell_id]
        train_samples = []
        for r in train_records:
            train_samples += make_windows(r, window=args.window, horizon=args.horizon)
        val_samples = make_windows(held_out, window=args.window, horizon=args.horizon)

        if not train_samples or not val_samples:
            print(f"Skipping {held_out.cell_id}: insufficient windows")
            continue

        print(f"\n=== held out: {held_out.cell_id} ({len(val_samples)} windows) ===")

        for arch in archs:
            _, _, _, pred, y_true = train_one_fold(
                train_records,
                val_samples,
                n_features,
                n_static,
                arch=arch,
                window=args.window,
                horizon=args.horizon,
            )
            m = metrics(y_true, pred)
            loco_results[arch][held_out.cell_id] = m
            loco_predictions[arch][held_out.cell_id] = {
                "target_cycle": [s.target_cycle for s in val_samples],
                "true_soh_pct": [round(float(v), 4) for v in y_true],
                "predicted_soh_pct": [round(float(v), 4) for v in pred],
            }
            print(
                f"  {arch:7s} MAE={m['mae']:.2f}  RMSE={m['rmse']:.2f}  "
                f"R2={m['r2']:.4f}  max_err={m['max_err']:.2f}"
            )

        rf_pred, y_true_rf = rf_baseline(train_samples, val_samples)
        rf_m = metrics(y_true_rf, rf_pred)
        loco_results["rf"][held_out.cell_id] = rf_m

        naive_pred, y_true_naive = naive_persistence(val_samples)
        naive_m = metrics(y_true_naive, naive_pred)
        loco_results["naive"][held_out.cell_id] = naive_m

        print(f"  {'naive':7s} MAE={naive_m['mae']:.2f}  RMSE={naive_m['rmse']:.2f}  R2={naive_m['r2']:.4f}  max_err={naive_m['max_err']:.2f}")
        print(f"  {'rf':7s} MAE={rf_m['mae']:.2f}  RMSE={rf_m['rmse']:.2f}  R2={rf_m['r2']:.4f}  max_err={rf_m['max_err']:.2f}")

    def agg(res):
        """Aggregate per-cell metrics into fleet-level means.

        R2 is averaged with nanmean because a cell whose targets are
        perfectly flat has an undefined R2 (see `metrics`); pooled_r2 is the
        more robust headline, computed once over every cell's predictions
        concatenated together rather than as a mean of per-cell R2s (which
        would let one narrow-span cell dominate).
        """
        maes = [v["mae"] for v in res.values()]
        rmses = [v["rmse"] for v in res.values()]
        r2s = [v["r2"] for v in res.values()]
        return {
            "mean_mae": float(np.mean(maes)),
            "mean_rmse": float(np.mean(rmses)),
            "mean_r2": float(np.nanmean(r2s)),
        }

    summary = {k: agg(v) for k, v in loco_results.items() if v}

    # Pooled R2: concatenate every held-out cell's predictions and score once.
    for arch in archs:
        all_true, all_pred = [], []
        for cell_id, p in loco_predictions[arch].items():
            all_true += p["true_soh_pct"]
            all_pred += p["predicted_soh_pct"]
        if all_true:
            summary[arch]["pooled_r2"] = float(r2_score(np.array(all_true), np.array(all_pred)))

    summary["per_cell"] = loco_results
    summary["archs"] = archs
    summary["arch_labels"] = {a: ARCH_LABELS.get(a, a.upper()) for a in archs}

    print("\n=== LOCO summary (mean over cells) ===")
    for k in list(archs) + ["naive", "rf"]:
        if k in summary:
            s = summary[k]
            pooled = f"  pooled_R2={s['pooled_r2']:.4f}" if "pooled_r2" in s else ""
            print(f"  {k:7s} MAE={s['mean_mae']:.4f}  RMSE={s['mean_rmse']:.4f}  mean_R2={s['mean_r2']:.4f}{pooled}")

    with open(out_dir / "loco_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    with open(out_dir / "loco_predictions.json", "w") as f:
        json.dump(loco_predictions, f)
    print(f"\nSaved out-of-sample LOCO predictions for {len(archs)} architecture(s) to {out_dir / 'loco_predictions.json'}")

    # Final production models: train on ALL cells (no held-out), for
    # deployment. Uses a fixed epoch budget matched to the LOCO folds'
    # typical early-stop point, since there is no held-out cell left to
    # validate against here.
    all_samples = []
    for r in records:
        all_samples += make_windows(r, window=args.window, horizon=args.horizon)
    seq_norm = Normalizer.fit(all_samples)
    static_norm = StaticNormalizer.fit(all_samples)
    x_seq, x_static, y, last_soh = samples_to_tensors(all_samples, seq_norm, static_norm)

    for arch in archs:
        cfg = LSTMConfig(n_features=n_features, n_static=n_static, soh_feature_idx=SOH_IDX, arch=arch)
        model = SOHRecurrent(cfg)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        loss_fn = nn.SmoothL1Loss()
        for epoch in range(150):
            model.train()
            opt.zero_grad()
            pred = model(x_seq, x_static, last_soh)
            loss = loss_fn(pred, y)
            loss.backward()
            opt.step()

        # The LSTM keeps the historical filename so existing deployments and
        # the ONNX exporter's default path continue to resolve; the others
        # are suffixed by architecture.
        fname = "soh_lstm.pt" if arch == "lstm" else f"soh_{arch}.pt"
        torch.save({
            "model_state": model.state_dict(),
            "config": cfg.__dict__,
            "seq_mean": seq_norm.mean,
            "seq_std": seq_norm.std,
            "static_mean": static_norm.mean,
            "static_std": static_norm.std,
            "window": args.window,
            "horizon": args.horizon,
            "feature_keys": SEQUENCE_FEATURE_KEYS,
            "arch": arch,
        }, out_dir / fname)
        print(f"Production {arch:7s} trained on all {len(records)} cells, "
              f"{len(all_samples)} windows, final loss={loss.item():.4f} -> {out_dir / fname}")


if __name__ == "__main__":
    main()
