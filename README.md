# VL Coin-Cell SOH Predictor

A small, fully client-side web app that predicts battery State of Health
(SOH) for Panasonic VL-series coin cells, using an LSTM trained on real
cycling data collected on a Neware tester.

**Live demo:** deploy `web/` to GitHub Pages (see below) — no server needed,
everything runs in the browser.

## What's in this repo

```
vl-soh-app/
├── data/
│   ├── raw/            # Neware xlsx exports (gitignored — not published)
│   └── processed/      # per-cell .npz caches (SOH labels + features + curves)
├── src/
│   ├── data/
│   │   ├── parse_neware.py      # xlsx -> cleaned DataFrame
│   │   ├── soh_features.py      # SOH labeling + feature extraction
│   │   ├── build_npz.py         # xlsx -> standardized .npz cache
│   │   └── export_web_json.py   # npz -> compact JSON for the web app
│   └── models/
│       ├── dataset.py           # sliding-window sample construction
│       ├── lstm_soh.py          # residual LSTM architecture
│       ├── train.py             # LOCO cross-validated training + RF baseline
│       └── export_onnx.py       # PyTorch -> ONNX for browser inference
├── models/              # trained soh_lstm.pt + loco_results.json
└── web/                 # the static app — deploy this folder as-is
    ├── index.html
    ├── app.js
    ├── styles.css
    ├── vendor/           # vendored Chart.js + onnxruntime-web (no CDN dependency)
    ├── model/            # soh_lstm.onnx + metadata (normalization stats etc.)
    └── data/             # per-cell JSON + catalog.json
```

## Data

Six Panasonic VL coin cells (VL1220 / VL2020 form factors), cycled at
0.1C/0.2C, at room temperature and 40°C:

| Cell | Status |
|---|---|
| VL1220-0.2C-RT | ✅ processed |
| VL1220-0.2C-40 | ✅ processed |
| VL2020-0.2C-RT | ✅ processed |
| VL2020-0.2C-40 | ✅ processed |
| VL1220-0.1C-RT | ⏳ not yet processed — source xlsx exceeded the data-transfer size limit during retrieval |
| VL2020-0.1C-RT | ⏳ not yet processed — same reason |

The two 0.1C-RT files are larger (slower C-rate → longer test → more rows)
and need to be added directly (e.g. dropped into `data/raw/`) — then re-run
the pipeline below to include them in training.

Each Neware export has columns: `Battery type, DataPoint, Cycle Index, Step
Index, Step Type, Elapsed Time(s), Temperature, Time, Total Time,
Current(A), Voltage(V), Capacity(Ah), SOC/DOD(%)`, sampled roughly every
30 seconds across 100 full charge/discharge cycles per cell.

### SOH definition

These coin cells show a real capacity **increase** during the first several
cycles (formation/break-in) before fade begins — using cycle 1 as the SOH
reference (common for larger-format cells) would understate early SOH. So:

```
C_ref      = max discharge capacity in the first 10 cycles (post-formation peak)
SOH(cycle) = discharge_capacity(cycle) / C_ref × 100
```

An 80%-SOH "end of life" point is also computed (first cycle where SOH drops
to and stays below 80%). These coin cells fade fast — all four processed
cells hit 80% SOH within 15–23 cycles, faster at 40°C than at room
temperature.

## Pipeline (regenerating everything from raw xlsx)

```bash
pip install -r requirements.txt

cd src/data
python build_npz.py ../../data/raw/*.xlsx --out-dir ../../data/processed
python export_web_json.py --npz-dir ../../data/processed --out-dir ../../web/data

cd ../models
python train.py --npz-dir ../../data/processed --out-dir ../../models
python export_onnx.py
```

## Model

A small **residual LSTM** (16 hidden units) predicts next-cycle SOH from a
10-cycle window of per-cycle summary features (SOH, coulombic efficiency,
step durations, temperature, voltage stats), conditioned on static cell
metadata (C-rate, temperature, form factor). The output is `last_observed_SOH
+ learned_correction` rather than an absolute value — with only a handful of
training cells, a model predicting SOH from scratch overfits badly to the
training cells' specific trajectories; predicting the *correction* to the
last known value is a much smaller, more transferable quantity, and is what
made the model actually beat a naive persistence baseline (see below).

### Validation: leave-one-cell-out (LOCO)

With so few cells, a random train/test split on sliding windows would leak
information between overlapping windows of the same cell. Every reported
number below holds out one entire cell — the model never sees any cycle
from it during training or model selection — then evaluates on that cell.

| Method | Mean MAE (SOH %) | Mean RMSE |
|---|---|---|
| Naive persistence (predict SOH unchanged) | 1.11 | 1.74 |
| Random Forest (flattened window + static features) | 2.12 | 2.93 |
| **Residual LSTM** | **0.77** | **1.30** |

The LSTM beats both baselines on every held-out cell. See
`models/loco_results.json` for the full per-cell breakdown.

## Web app

Fully static, no backend: `web/` can be served as-is (GitHub Pages, Netlify,
or `python -m http.server` locally). All JS/WASM dependencies are vendored
in `web/vendor/` — no CDN calls, no external network requests, all
inference and data stay on-device.

- **Chart.js 4.4.4** for plotting SOH curves and cycle-detail voltage traces.
- **ONNX Runtime Web 1.18.0** (WASM backend) for in-browser LSTM inference.

### Deploying to GitHub Pages

```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin <your-repo-url>
git push -u origin main
```

Then in the repo settings, enable GitHub Pages with source = `main` branch,
folder = `/web`. (Or use a `gh-pages` branch / GitHub Action if you prefer
keeping `web/` un-nested at the Pages root.)

## Extending to NASA / CALCE / Oxford datasets

The `.npz` schema here (documented at the top of `build_npz.py`) is
self-contained and was designed from scratch, since the existing NASA/CALCE/
Oxford npz caches weren't available to inspect during development. To add
those datasets to this app:

1. Write an adapter that reads each dataset's native format (or existing
   npz cache) and re-emits the same field names used here (`cycles`,
   `soh_pct`, `dchg_voltage`, etc.) — or extend `web/app.js`'s data loading
   to handle multiple schemas behind a common interface.
2. Add each cell to `web/data/catalog.json` and drop its JSON file next to
   the VL ones.
3. Retrain (`train.py`) including the new cells for a model that
   generalizes across chemistries/form-factors, or keep per-dataset models
   if the fade behavior is too different to share one model usefully.
