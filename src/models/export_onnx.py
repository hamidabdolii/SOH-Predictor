"""
Export the trained SOH-LSTM to ONNX for fully client-side inference in the
web app (onnxruntime-web), plus a small JSON sidecar with everything the
frontend needs to reproduce preprocessing (normalization stats, feature
order, window size) since ONNX only captures the model's forward pass.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from lstm_soh import SOHLSTM, LSTMConfig


def export(model_path: str, out_dir: str):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    cfg = LSTMConfig(**ckpt["config"])
    model = SOHLSTM(cfg)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    window = ckpt["window"]
    n_features = cfg.n_features
    n_static = cfg.n_static

    dummy_seq = torch.randn(1, window, n_features)
    dummy_static = torch.randn(1, n_static)
    dummy_last_soh = torch.randn(1)

    onnx_path = out_dir / "soh_lstm.onnx"
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
    print(f"Exported ONNX model to {onnx_path}")

    meta = {
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
    meta_path = out_dir / "soh_lstm_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Exported metadata to {meta_path}")

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
    print(f"Max diff between PyTorch and ONNX outputs on random input: {max_diff:.2e}")
    assert max_diff < 1e-3, "ONNX export mismatch too large!"
    print("ONNX export verified OK")


if __name__ == "__main__":
    export("../../models/soh_lstm.pt", "../../web/model")
