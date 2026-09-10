"""
Export the trained SOH models to ONNX for fully client-side inference in the
web app (onnxruntime-web), plus a small JSON sidecar with everything the
frontend needs to reproduce preprocessing (normalization stats, feature
order, window size) since ONNX only captures the model's forward pass.

Every trained architecture (LSTM / GRU / Simple RNN / Bi-LSTM) is exported,
so the dashboard's model picker can switch between them live in the browser.
The LSTM keeps the unsuffixed `soh_lstm.onnx` / `soh_lstm_meta.json` names it
has always had; the others are suffixed by architecture. A `models.json`
index lists what is available, which is what the frontend reads.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from lstm_soh import SOHRecurrent, LSTMConfig, ARCH_CHOICES, ARCH_LABELS


def export_one(model_path: Path, out_dir: Path, arch: str) -> dict | None:
    """Export a single trained checkpoint to ONNX + metadata sidecar.

    Returns an index entry for models.json, or None if the checkpoint is
    missing (e.g. training was run with a subset of --archs).
    """
    if not model_path.exists():
        print(f"skip {arch}: no checkpoint at {model_path}")
        return None

    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    cfg = LSTMConfig(**ckpt["config"])
    model = SOHRecurrent(cfg)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    window = ckpt["window"]
    n_features = cfg.n_features
    n_static = cfg.n_static

    dummy_seq = torch.randn(1, window, n_features)
    dummy_static = torch.randn(1, n_static)
    dummy_last_soh = torch.randn(1)

    stem = "soh_lstm" if arch == "lstm" else f"soh_{arch}"
    onnx_path = out_dir / f"{stem}.onnx"
    torch.onnx.export(
        model,
        (dummy_seq, dummy_static, dummy_last_soh),
        str(onnx_path),
        input_names=["x_seq", "x_static", "raw_last_soh"],
        output_names=["soh_pred"],
        dynamic_axes={
            "x_seq": {0: "batch"},
            "x_static": {0: "batch"},
            "raw_last_soh": {0: "batch"},
            "soh_pred": {0: "batch"},
        },
        opset_version=17,
        dynamo=False,  # legacy TorchScript-based exporter: produces an older,
                        # more broadly-compatible IR version than the newer
                        # dynamo-based exporter (which emits IR v10, too new
                        # for some onnxruntime-web wasm builds).
    )

    meta = {
        "arch": arch,
        "label": ARCH_LABELS.get(arch, arch.upper()),
        "window": int(window),
        "horizon": int(ckpt["horizon"]),
        "feature_keys": list(ckpt["feature_keys"]),
        "soh_feature_idx": int(cfg.soh_feature_idx),
        "seq_mean": ckpt["seq_mean"].tolist(),
        "seq_std": ckpt["seq_std"].tolist(),
        "static_mean": ckpt["static_mean"].tolist(),
        "static_std": ckpt["static_std"].tolist(),
        "static_feature_order": ["c_rate_numeric", "temp_nominal_c", "is_vl2020_form_factor"],
    }
    meta_path = out_dir / f"{stem}_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    # sanity check: verify onnxruntime output matches pytorch output
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path))
    ort_out = sess.run(None, {
        "x_seq": dummy_seq.numpy(),
        "x_static": dummy_static.numpy(),
        "raw_last_soh": dummy_last_soh.numpy(),
    })[0]
    with torch.no_grad():
        torch_out = model(dummy_seq, dummy_static, dummy_last_soh).numpy()
    max_diff = np.abs(ort_out.flatten() - torch_out.flatten()).max()
    assert max_diff < 1e-3, f"{arch}: ONNX export mismatch too large ({max_diff:.2e})!"

    n_params = sum(p.numel() for p in model.parameters())
    print(f"OK  {arch:7s} -> {onnx_path.name} ({n_params:,} params, ONNX/PyTorch max diff {max_diff:.1e})")

    return {
        "arch": arch,
        "label": ARCH_LABELS.get(arch, arch.upper()),
        "onnx": f"{stem}.onnx",
        "meta": f"{stem}_meta.json",
        "n_params": int(n_params),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-dir", default="../../models")
    ap.add_argument("--out-dir", default="../../web/model")
    ap.add_argument("--archs", default=",".join(ARCH_CHOICES))
    args = ap.parse_args()

    models_dir = Path(args.models_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    index = []
    for arch in [a.strip() for a in args.archs.split(",") if a.strip()]:
        fname = "soh_lstm.pt" if arch == "lstm" else f"soh_{arch}.pt"
        entry = export_one(models_dir / fname, out_dir, arch)
        if entry:
            index.append(entry)

    with open(out_dir / "models.json", "w") as f:
        json.dump(index, f, indent=2)
    print(f"\nWrote model index with {len(index)} architecture(s) to {out_dir / 'models.json'}")


if __name__ == "__main__":
    main()
