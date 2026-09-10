// VL Coin-Cell SOH Predictor — client-side app.
// Loads pre-computed per-cell JSON (data/<cell_id>.json), runs the ONNX LSTM
// in-browser for next-cycle SOH prediction, and renders comparison charts.

const DATA_BASE = "data";
const MODEL_BASE = "model";

let catalog = [];
let cellDataCache = {}; // cell_id -> parsed JSON
let modelMeta = null;          // metadata for the CURRENTLY selected architecture
let ortSession = null;         // ONNX session for the CURRENTLY selected architecture
let selectedCellId = null;
let locoResults = null;
let modelIndex = [];           // [{arch, label, onnx, meta, n_params}, ...]
let selectedArch = "lstm";     // which architecture the UI is currently showing
const ortSessionCache = {};    // arch -> InferenceSession (models are tiny; keep them)
const modelMetaCache = {};     // arch -> parsed metadata JSON

let sohChart = null;
let capacityChart = null;
let tempTrendChart = null;
let ceChart = null;
let curveVoltageChart = null;
let curveCurrentChart = null;
let curveTempChart = null;
let compareChart = null;
let trajectoryChart = null;
let trajectoryRunToken = 0; // lets a new cell selection cancel an in-flight animated reveal

const CHART_COLORS = ["#5eb1ff", "#ff9f5e", "#4ade80", "#f87171", "#c084fc", "#fbbf24"];

// Two color families for the cross-cell comparison chart: cool blues for VL
// coin cells, warm oranges for Oxford pouch cells, so the two chemistries/
// form-factors read as visually distinct groups rather than one arbitrary
// palette cycling across 12 lines.
const VL_COMPARE_COLORS = ["#5eb1ff", "#7dc4ff", "#3d8fe0", "#a8d8ff"];
const OXFORD_COMPARE_COLORS = ["#ff9f5e", "#ffb37a", "#e0803d", "#ffc79a", "#ff8a3d", "#ffab6b", "#d97227", "#ffcc9e"];

function compareColorFor(cell, indexWithinGroup) {
  if (cell.form_factor === "OXFORD") return OXFORD_COMPARE_COLORS[indexWithinGroup % OXFORD_COMPARE_COLORS.length];
  return VL_COMPARE_COLORS[indexWithinGroup % VL_COMPARE_COLORS.length];
}

// Shared Chart.js visual defaults so every chart in the app reads as one
// coherent, legible system instead of library defaults bolted on ad hoc.
Chart.defaults.color = "#aab4cc";
Chart.defaults.font.family = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif";
Chart.defaults.font.size = 12;
Chart.defaults.borderColor = "rgba(255,255,255,0.08)";

function gridOpts() {
  return {
    grid: { color: "rgba(255,255,255,0.07)", tickBorderDash: [2, 2] },
    ticks: { color: "#aab4cc" },
    title: { color: "#c7cfe2", font: { weight: "600", size: 12 } },
  };
}

async function fetchJSON(path) {
  const res = await fetch(path);
  if (!res.ok) throw new Error(`Failed to fetch ${path}: ${res.status}`);
  return res.json();
}

async function loadCatalog() {
  catalog = await fetchJSON(`${DATA_BASE}/catalog.json`);
}

async function loadLocoResults() {
  try {
    locoResults = await fetchJSON(`${DATA_BASE}/loco_results.json`);
  } catch (e) {
    console.warn("No loco_results.json found; skipping fleet stats / LOCO table.", e);
    locoResults = null;
  }
}

function renderFleetStats() {
  const el = document.getElementById("fleet-stats");
  if (!el) return;
  const nCells = catalog.length;
  const nDatasets = new Set(catalog.map(datasetGroupOf)).size;

  const perCell = locoResults && locoResults.per_cell ? locoResults.per_cell[selectedArch] : null;
  let predictionCount = 0;
  if (perCell) {
    predictionCount = Object.values(perCell).reduce((sum, m) => sum + (m.n || 0), 0);
  }

  const archSummary = locoResults ? locoResults[selectedArch] : null;
  const naiveMae = locoResults && locoResults.naive ? locoResults.naive.mean_mae : null;

  el.innerHTML =
    statTile("layers", "Cells in fleet", `${nCells}`, "accent") +
    statTile("flask", "Datasets", `${nDatasets}`) +
    statTile("target", "Out-of-sample predictions (LOCO)", predictionCount.toLocaleString()) +
    (archSummary
      ? statTile("brain", `Mean LOCO MAE (${archLabel(selectedArch)})`, `${archSummary.mean_mae.toFixed(2)} pts`, "good", "good")
      : "") +
    (archSummary && archSummary.pooled_r2 !== undefined
      ? statTile("r2", "Pooled R²", archSummary.pooled_r2.toFixed(4), "good", "good")
      : "") +
    (naiveMae !== null
      ? statTile("scale", "vs. naive persistence", `${naiveMae.toFixed(2)} pts`)
      : "");
}

/** Fleet-level comparison of every architecture plus the two baselines. */
function renderArchTable() {
  const table = document.getElementById("arch-table");
  if (!table || !locoResults) return;
  const tbody = table.querySelector("tbody");
  tbody.innerHTML = "";

  const archs = locoResults.archs || ["lstm"];
  const rows = [
    ...archs.map((a) => ({ key: a, label: archLabel(a), isArch: true })),
    { key: "naive", label: "Naive persistence", isArch: false },
    { key: "rf", label: "Random Forest", isArch: false },
  ];

  // Best (lowest) mean MAE among the neural architectures, for highlighting.
  const bestMae = Math.min(...archs.map((a) => (locoResults[a] ? locoResults[a].mean_mae : Infinity)));

  rows.forEach(({ key, label, isArch }) => {
    const s = locoResults[key];
    if (!s) return;
    const entry = modelIndex.find((m) => m.arch === key);
    const isBest = isArch && Math.abs(s.mean_mae - bestMae) < 1e-9;
    const isSelected = isArch && key === selectedArch;
    const tr = document.createElement("tr");
    if (isSelected) tr.className = "row-selected";
    tr.innerHTML = `
      <td>${label}${isBest ? ' <span class="best-badge">best</span>' : ""}${isSelected ? ' <span class="shown-badge">shown</span>' : ""}</td>
      <td class="num ${isArch ? "good-text" : "muted"}">${s.mean_mae.toFixed(3)}</td>
      <td class="num ${isArch ? "" : "muted"}">${s.mean_rmse.toFixed(3)}</td>
      <td class="num ${isArch ? "" : "muted"}">${s.mean_r2 !== undefined ? s.mean_r2.toFixed(4) : "–"}</td>
      <td class="num ${isArch ? "" : "muted"}">${s.pooled_r2 !== undefined ? s.pooled_r2.toFixed(4) : "–"}</td>
      <td class="num muted">${entry && entry.n_params ? entry.n_params.toLocaleString() : "–"}</td>
    `;
    tbody.appendChild(tr);
  });
}

function renderLocoTable() {
  const table = document.getElementById("loco-table");
  if (!table || !locoResults) return;
  const tbody = table.querySelector("tbody");
  tbody.innerHTML = "";
  const archRes = locoResults.per_cell[selectedArch] || {};
  const naiveRes = locoResults.per_cell.naive || {};
  const rfRes = locoResults.per_cell.rf || {};

  catalog.forEach((cell) => {
    const id = cell.cell_id;
    const m = archRes[id];
    const naive = naiveRes[id];
    const rf = rfRes[id];
    if (!m) return;
    const r2Class = m.r2 === undefined || m.r2 === null || Number.isNaN(m.r2)
      ? "muted"
      : m.r2 >= 0.99 ? "good-text" : m.r2 >= 0.95 ? "warn-text" : "bad-text";
    const r2Text = m.r2 === undefined || m.r2 === null || Number.isNaN(m.r2) ? "–" : m.r2.toFixed(4);
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${cell.cell_id}</td>
      <td>${datasetGroupOf(cell)}</td>
      <td class="num good-text">${m.mae.toFixed(2)}</td>
      <td class="num">${m.rmse.toFixed(2)}</td>
      <td class="num ${r2Class}">${r2Text}</td>
      <td class="num muted">${m.max_err !== undefined ? m.max_err.toFixed(2) : "–"}</td>
      <td class="num muted">${m.soh_span !== undefined ? m.soh_span.toFixed(1) : "–"}</td>
      <td class="num muted">${naive ? naive.mae.toFixed(2) : "–"}</td>
      <td class="num muted">${rf ? rf.mae.toFixed(2) : "–"}</td>
    `;
    tbody.appendChild(tr);
  });
}

function updateAboutModelText() {
  const el = document.getElementById("about-model-text");
  if (!el || !locoResults) return;
  const naiveMae = locoResults.naive.mean_mae;
  const rfMae = locoResults.rf.mean_mae;
  const nCells = catalog.length;
  const archs = locoResults.archs || ["lstm"];
  const best = archs.reduce((a, b) => (locoResults[a].mean_mae <= locoResults[b].mean_mae ? a : b));
  const bestS = locoResults[best];
  el.innerHTML =
    `Four small residual recurrent models &mdash; ${archs.map(archLabel).join(", ")} &mdash; each predict next-cycle SOH from a ` +
    `10-cycle window of summary features (capacity, coulombic efficiency, step durations, temperature, voltage), conditioned on ` +
    `C-rate/temperature/form-factor. Each predicts a <em>correction</em> to the last observed SOH rather than an absolute value, which is ` +
    `what lets them generalize from so few cells. All four were validated with leave-one-cell-out cross-validation across all ${nCells} cells ` +
    `(VL coin cells and Oxford pouch cells) under an identical protocol. Best of the four is <strong>${archLabel(best)}</strong> at ` +
    `<strong>${bestS.mean_mae.toFixed(2)} pts MAE</strong> and pooled R² of <strong>${(bestS.pooled_r2 ?? 0).toFixed(4)}</strong> on cells never ` +
    `seen in training &mdash; every architecture beats both the naive persistence baseline (${naiveMae.toFixed(2)} pts) and the Random Forest ` +
    `baseline (${rfMae.toFixed(2)} pts). All inference runs locally in your browser via ONNX Runtime Web; no data leaves your device.`;
}

async function loadCellData(cellId) {
  if (cellDataCache[cellId]) return cellDataCache[cellId];
  const data = await fetchJSON(`${DATA_BASE}/${cellId}.json`);
  cellDataCache[cellId] = data;
  return data;
}

async function loadModelIndex() {
  try {
    modelIndex = await fetchJSON(`${MODEL_BASE}/models.json`);
  } catch (e) {
    // Fall back to the single LSTM model if no index is present (older build).
    console.warn("No models.json found; falling back to the LSTM only.", e);
    modelIndex = [{ arch: "lstm", label: "LSTM", onnx: "soh_lstm.onnx", meta: "soh_lstm_meta.json" }];
  }
  if (!modelIndex.some((m) => m.arch === selectedArch)) {
    selectedArch = modelIndex.length ? modelIndex[0].arch : "lstm";
  }
}

async function loadModel(arch = selectedArch) {
  // Single-threaded WASM: GitHub Pages doesn't set the COOP/COEP headers
  // required for SharedArrayBuffer, so multi-threaded WASM execution isn't
  // available there. The models are tiny (16 hidden units), so single-thread
  // CPU inference is effectively instant regardless.
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.wasmPaths = "vendor/";

  const entry = modelIndex.find((m) => m.arch === arch) || modelIndex[0];
  if (!modelMetaCache[arch]) {
    modelMetaCache[arch] = await fetchJSON(`${MODEL_BASE}/${entry.meta}`);
  }
  if (!ortSessionCache[arch]) {
    ortSessionCache[arch] = await ort.InferenceSession.create(`${MODEL_BASE}/${entry.onnx}`);
  }
  modelMeta = modelMetaCache[arch];
  ortSession = ortSessionCache[arch];
}

function archLabel(arch) {
  const entry = modelIndex.find((m) => m.arch === arch);
  return entry ? entry.label : arch.toUpperCase();
}

/** The LOCO predictions for one cell under the currently selected arch. */
function locoFor(data, arch = selectedArch) {
  if (data.loco_by_arch && data.loco_by_arch[arch]) return data.loco_by_arch[arch];
  // Back-compat with JSON exported before the multi-arch change.
  if (arch === "lstm" && data.loco) return data.loco;
  return null;
}

function renderModelPicker() {
  const host = document.getElementById("model-picker-options");
  if (!host) return;
  host.innerHTML = "";
  modelIndex.forEach((m) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "model-chip" + (m.arch === selectedArch ? " selected" : "");
    btn.dataset.arch = m.arch;
    const params = m.n_params ? `<span class="model-chip-params">${m.n_params.toLocaleString()} params</span>` : "";
    const mae = locoResults && locoResults[m.arch] ? `<span class="model-chip-mae">MAE ${locoResults[m.arch].mean_mae.toFixed(3)}</span>` : "";
    btn.innerHTML = `<span class="model-chip-name">${m.label}</span>${mae}${params}`;
    btn.addEventListener("click", () => selectArch(m.arch));
    host.appendChild(btn);
  });
}

async function selectArch(arch) {
  if (arch === selectedArch) return;
  selectedArch = arch;
  renderModelPicker();
  await loadModel(arch);

  // Everything that depends on which architecture is selected.
  renderFleetStats();
  renderLocoTable();
  renderArchTable();
  updateAboutModelText();
  const label = document.getElementById("loco-table-arch-label");
  if (label) label.textContent = archLabel(arch);
  const predLabel = document.getElementById("predict-arch-label");
  if (predLabel) predLabel.textContent = archLabel(arch);

  if (selectedCellId && cellDataCache[selectedCellId]) {
    setupTrajectoryPanel(cellDataCache[selectedCellId]);
    document.getElementById("predict-result").innerHTML = "";
  }
}

function icon(name, cls = "") {
  return `<svg class="icon ${cls}"><use href="#icon-${name}"/></svg>`;
}

function statTile(iconName, label, value, valueClass = "", badgeClass = "") {
  return `
    <div class="stat">
      <div class="icon-badge ${badgeClass}">${icon(iconName)}</div>
      <div class="body">
        <div class="label">${label}</div>
        <div class="value ${valueClass}">${value}</div>
      </div>
    </div>`;
}

function datasetGroupOf(cell) {
  if (cell.form_factor === "OXFORD") return "Oxford (pouch)";
  return "VL coin cells";
}

function renderCellGrid() {
  const grid = document.getElementById("cell-grid");
  grid.innerHTML = "";

  const groups = new Map();
  catalog.forEach((cell) => {
    const g = datasetGroupOf(cell);
    if (!groups.has(g)) groups.set(g, []);
    groups.get(g).push(cell);
  });

  groups.forEach((cells, groupName) => {
    const groupWrap = document.createElement("div");
    groupWrap.className = "cell-group";
    groupWrap.innerHTML = `<h3 class="cell-group-title">${groupName} <span class="cell-group-count">${cells.length} cell${cells.length === 1 ? "" : "s"}</span></h3>`;
    const row = document.createElement("div");
    row.className = "cell-grid";
    cells.forEach((cell) => {
      const card = document.createElement("div");
      card.className = "cell-card";
      card.dataset.cellId = cell.cell_id;
      card.innerHTML = `
        <h4>${icon("battery", "icon-accent")} ${cell.cell_id}</h4>
        <div class="meta">${cell.form_factor} &middot; ${cell.c_rate} &middot; ${cell.temp_condition === "RT" ? "Room Temp" : cell.temp_condition + "&deg;C"}</div>
        <div class="meta">${icon("cycle", "icon-sm")} ${cell.n_cycles} cycles tested</div>
        <span class="eol-badge">${icon("target", "icon-sm")} 80% SOH @ cycle ${cell.eol_cycle_80pct > 0 ? cell.eol_cycle_80pct : "n/a"}</span>
      `;
      card.addEventListener("click", () => selectCell(cell.cell_id));
      row.appendChild(card);
    });
    groupWrap.appendChild(row);
    grid.appendChild(groupWrap);
  });
}

async function selectCell(cellId) {
  selectedCellId = cellId;
  document.querySelectorAll(".cell-card").forEach((c) => {
    c.classList.toggle("selected", c.dataset.cellId === cellId);
  });

  const data = await loadCellData(cellId);

  document.getElementById("trajectory-panel").hidden = false;
  document.getElementById("overview-panel").hidden = false;
  document.getElementById("predict-panel").hidden = false;
  document.getElementById("curves-panel").hidden = false;

  setupTrajectoryPanel(data);
  renderSOHChart(data);
  renderCapacityChart(data);
  renderTempTrendChart(data);
  renderCEChart(data);
  renderStats(data);
  setupCurveSlider(data);
  document.getElementById("predict-result").innerHTML = "";

  const maxCycle = data.cycles[data.cycles.length - 1];
  const slider = document.getElementById("cycle-slider");
  const minCycle = data.cycles[0];
  slider.min = minCycle;
  slider.max = maxCycle - 1;
  slider.value = Math.min(minCycle + 10, maxCycle - 1);
  document.getElementById("cycle-label").textContent = slider.value;
}

function renderSOHChart(data) {
  const ctx = document.getElementById("soh-chart").getContext("2d");
  if (sohChart) sohChart.destroy();
  sohChart = new Chart(ctx, {
    type: "line",
    data: {
      labels: data.cycles,
      datasets: [
        {
          label: "SOH (%)",
          data: data.soh_pct,
          borderColor: CHART_COLORS[0],
          backgroundColor: "transparent",
          borderWidth: 2.25,
          pointRadius: 0,
          tension: 0.15,
        },
        {
          label: "80% EOL threshold",
          data: data.cycles.map(() => 80),
          borderColor: "#f87171",
          borderDash: [6, 4],
          pointRadius: 0,
          borderWidth: 1.5,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: { title: { display: true, text: "Cycle" }, ...gridOpts() },
        y: { title: { display: true, text: "SOH (%)" }, min: 0, max: 105, ...gridOpts() },
      },
      plugins: { legend: { position: "bottom", labels: { usePointStyle: true, boxWidth: 8, boxHeight: 8 } } },
    },
  });
}

function renderCapacityChart(data) {
  const ctx = document.getElementById("capacity-chart").getContext("2d");
  if (capacityChart) capacityChart.destroy();
  capacityChart = new Chart(ctx, {
    type: "line",
    data: {
      labels: data.cycles,
      datasets: [
        {
          label: "Discharge capacity (mAh)",
          data: data.discharge_capacity_mah,
          borderColor: CHART_COLORS[3],
          backgroundColor: "transparent",
          borderWidth: 2,
          pointRadius: 0,
          tension: 0.15,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: { title: { display: true, text: "Cycle" }, ...gridOpts() },
        y: { title: { display: true, text: "Capacity (mAh)" }, beginAtZero: true, ...gridOpts() },
      },
      plugins: { legend: { display: false } },
    },
  });
}

function renderTempTrendChart(data) {
  const ctx = document.getElementById("temp-trend-chart").getContext("2d");
  if (tempTrendChart) tempTrendChart.destroy();
  tempTrendChart = new Chart(ctx, {
    type: "line",
    data: {
      labels: data.cycles,
      datasets: [
        {
          label: "Mean temperature (°C)",
          data: data.mean_temperature_c,
          borderColor: CHART_COLORS[5],
          backgroundColor: "transparent",
          borderWidth: 2,
          pointRadius: 0,
          tension: 0.15,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: { title: { display: true, text: "Cycle" }, ...gridOpts() },
        y: { title: { display: true, text: "Temperature (°C)" }, ...gridOpts() },
      },
      plugins: { legend: { display: false } },
    },
  });
}

function renderCEChart(data) {
  const ctx = document.getElementById("ce-chart").getContext("2d");
  if (ceChart) ceChart.destroy();
  ceChart = new Chart(ctx, {
    type: "line",
    data: {
      labels: data.cycles,
      datasets: [
        {
          label: "Coulombic efficiency",
          data: data.coulombic_efficiency,
          borderColor: CHART_COLORS[2],
          backgroundColor: "transparent",
          borderWidth: 2,
          pointRadius: 0,
          tension: 0.15,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: { title: { display: true, text: "Cycle" }, ...gridOpts() },
        y: { title: { display: true, text: "CE (ratio)" }, ...gridOpts() },
      },
      plugins: { legend: { display: false } },
    },
  });
}

function renderStats(data) {
  const statsRow = document.getElementById("cell-stats");
  const cRef = data.c_ref_ah;
  const eol = data.eol_cycle_80pct;
  const finalSoh = data.soh_pct[data.soh_pct.length - 1];
  statsRow.innerHTML =
    statTile("layers", "Reference capacity", `${(cRef * 1000).toFixed(2)} mAh`) +
    statTile("target", "80% EOL cycle", eol > 0 ? eol : "n/a") +
    statTile("battery", `Final SOH (cycle ${data.cycles[data.cycles.length - 1]})`, `${finalSoh.toFixed(1)}%`);
}

// --- Predicted vs actual SOH trajectory (out-of-sample, leave-one-cell-out) ---

function setupTrajectoryPanel(data) {
  trajectoryRunToken++; // invalidate any in-flight animation from a previous cell
  const btn = document.getElementById("run-trajectory-btn");
  const statsRow = document.getElementById("trajectory-stats");
  statsRow.innerHTML = "";
  renderTrajectoryChart(data, 0); // start empty, full true curve only

  if (!locoFor(data)) {
    btn.disabled = true;
    btn.innerHTML = `${icon("target")} Not available for this cell`;
    statsRow.innerHTML = `<p class="hint">This cell wasn't part of the leave-one-cell-out validation set for the ${archLabel(selectedArch)} model, so no genuine out-of-sample predictions are available for it yet.</p>`;
    return;
  }

  btn.disabled = false;
  btn.innerHTML = `${icon("play")} Run full prediction`;
  btn.onclick = () => runTrajectoryAnimation(data);
}

async function runTrajectoryAnimation(data) {
  const myToken = ++trajectoryRunToken;
  const btn = document.getElementById("run-trajectory-btn");
  const loco = locoFor(data);
  if (!loco) return;
  const n = loco.target_cycle.length;

  btn.disabled = true;
  btn.innerHTML = `${icon("clock")} Predicting...`;

  // Animate the reveal cycle-by-cycle so it reads as the model "walking
  // forward through time" rather than a static image — while every value
  // shown is a real, precomputed out-of-sample prediction (see loco field).
  const stepMs = n > 60 ? 12 : 25;
  for (let shown = 1; shown <= n; shown++) {
    if (myToken !== trajectoryRunToken) return; // a different cell was selected mid-animation
    renderTrajectoryChart(data, shown);
    updateTrajectoryStats(data, shown);
    if (shown < n) await new Promise((r) => setTimeout(r, stepMs));
  }

  if (myToken !== trajectoryRunToken) return;
  btn.disabled = false;
  btn.innerHTML = `${icon("play")} Run again`;
}

/** Coefficient of determination, R2 = 1 - SS_res/SS_tot.
 *
 * Returns null when there is not enough spread in the observed values for
 * R2 to mean anything (fewer than 2 points, or every target identical) —
 * the caller shows a dash instead of a misleading number. Note R2 is
 * measured against the variance of the values shown *so far*, so during
 * the animated reveal it starts unstable and settles as the window widens.
 */
function rSquared(trueVals, predVals) {
  const n = trueVals.length;
  if (n < 2) return null;
  const mean = trueVals.reduce((a, b) => a + b, 0) / n;
  let ssRes = 0;
  let ssTot = 0;
  for (let i = 0; i < n; i++) {
    ssRes += (trueVals[i] - predVals[i]) ** 2;
    ssTot += (trueVals[i] - mean) ** 2;
  }
  if (ssTot < 1e-12) return null;
  return 1 - ssRes / ssTot;
}

function updateTrajectoryStats(data, shownCount) {
  const loco = locoFor(data);
  if (!loco) return;
  const trueVals = loco.true_soh_pct.slice(0, shownCount);
  const predVals = loco.predicted_soh_pct.slice(0, shownCount);
  const errors = trueVals.map((t, i) => Math.abs(t - predVals[i]));
  const mae = errors.reduce((a, b) => a + b, 0) / errors.length;
  const maxErr = Math.max(...errors);
  const rmse = Math.sqrt(errors.reduce((a, b) => a + b * b, 0) / errors.length);
  const r2 = rSquared(trueVals, predVals);

  const statsRow = document.getElementById("trajectory-stats");
  const maeGood = mae < 1.5;
  // R2 above 0.99 is the bar these models clear on well-behaved cells; the
  // amber band flags a cell where the fit is materially worse.
  const r2Class = r2 === null ? "" : r2 >= 0.99 ? "good" : r2 >= 0.95 ? "warn" : "bad";
  statsRow.innerHTML =
    statTile("target", "Predictions made", `${shownCount} / ${loco.target_cycle.length}`, "accent") +
    statTile("scale", "Mean absolute error", `${mae.toFixed(2)} pts`, maeGood ? "good" : "warn", maeGood ? "good" : "warn") +
    statTile("r2", "R² so far", r2 === null ? "–" : r2.toFixed(4), r2Class, r2Class) +
    statTile("trend", "RMSE", `${rmse.toFixed(2)} pts`) +
    statTile("bolt", "Max error so far", `${maxErr.toFixed(2)} pts`);
}

function renderTrajectoryChart(data, shownCount) {
  const ctx = document.getElementById("trajectory-chart").getContext("2d");
  if (trajectoryChart) trajectoryChart.destroy();

  // Visual convention deliberately mirrors the reference LOCO plot: a bold,
  // pale-gray line for the full true trajectory in the background, solid
  // blue dots for the real (out-of-sample) targets, and orange X markers for
  // the model's predictions on top — same encoding, cleaner rendering.
  const datasets = [
    {
      label: "True SOH (all cycles)",
      data: data.cycles.map((c, i) => ({ x: c, y: data.soh_pct[i] })),
      borderColor: "#7b879e",
      backgroundColor: "transparent",
      borderWidth: 2.5,
      pointRadius: 0,
      order: 3,
    },
  ];

  const loco = locoFor(data);
  if (loco) {
    const trueShown = loco.target_cycle.slice(0, shownCount).map((c, i) => ({ x: c, y: loco.true_soh_pct[i] }));
    const predShown = loco.target_cycle.slice(0, shownCount).map((c, i) => ({ x: c, y: loco.predicted_soh_pct[i] }));
    datasets.push(
      {
        label: "Actual SOH (out-of-sample target)",
        data: trueShown,
        borderColor: "#2f8fff",
        backgroundColor: "#2f8fff",
        showLine: false,
        pointRadius: 5,
        pointHoverRadius: 6,
        order: 1,
      },
      {
        label: `Predicted SOH — ${archLabel(selectedArch)} (leave-one-cell-out)`,
        data: predShown,
        borderColor: "#ff8a2b",
        backgroundColor: "#ff8a2b",
        showLine: false,
        pointStyle: "crossRot",
        pointRadius: 6,
        pointBorderWidth: 2.5,
        pointHoverRadius: 7,
        order: 0,
      }
    );
  }

  trajectoryChart = new Chart(ctx, {
    type: "line",
    data: { datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      parsing: false,
      animation: false,
      layout: { padding: { top: 8, right: 12 } },
      scales: {
        x: { type: "linear", title: { display: true, text: "Cycle" }, ...gridOpts() },
        y: { title: { display: true, text: "SOH (%)" }, min: 0, max: 105, ...gridOpts() },
      },
      plugins: {
        legend: {
          position: "bottom",
          labels: { usePointStyle: true, boxWidth: 8, boxHeight: 8, padding: 18, font: { size: 13, weight: "600" } },
        },
        tooltip: {
          backgroundColor: "#1a2136",
          borderColor: "#32405f",
          borderWidth: 1,
          padding: 10,
          titleFont: { weight: "700" },
          callbacks: {
            title: (items) => `Cycle ${items[0].parsed.x}`,
            label: (item) => `${item.dataset.label}: ${item.parsed.y.toFixed(2)}%`,
          },
        },
      },
    },
  });
}

// --- Cycle-detail curves (voltage / current / temperature), any cycle ---

function setupCurveSlider(data) {
  const slider = document.getElementById("curve-cycle-slider");
  const label = document.getElementById("curve-cycle-label");
  slider.min = 0;
  slider.max = data.cycles.length - 1;
  slider.value = 0;
  label.textContent = `Cycle ${data.cycles[0]}`;

  slider.oninput = () => {
    const idx = parseInt(slider.value, 10);
    const cyc = data.cycles[idx];
    label.textContent = `Cycle ${cyc}`;
    renderCurveCharts(data, cyc);
  };
  renderCurveCharts(data, data.cycles[0]);
}

function renderCurveCharts(data, cycle) {
  const curve = data.curves[String(cycle)];
  if (!curve) return;
  const n = curve.dchg_voltage.length;
  const labels = Array.from({ length: n }, (_, i) => i);

  const commonOpts = (yLabel) => ({
    responsive: true,
    maintainAspectRatio: false,
    scales: {
      x: { title: { display: true, text: "Normalized time within step" }, ...gridOpts() },
      y: { title: { display: true, text: yLabel }, ...gridOpts() },
    },
    plugins: { legend: { position: "bottom", labels: { usePointStyle: true, boxWidth: 8, boxHeight: 8 } } },
  });

  if (curveVoltageChart) curveVoltageChart.destroy();
  curveVoltageChart = new Chart(document.getElementById("curve-voltage-chart").getContext("2d"), {
    type: "line",
    data: {
      labels,
      datasets: [
        { label: "Discharge voltage (V)", data: curve.dchg_voltage, borderColor: CHART_COLORS[3], pointRadius: 0 },
        { label: "Charge voltage (V)", data: curve.chg_voltage, borderColor: CHART_COLORS[0], pointRadius: 0 },
      ],
    },
    options: commonOpts("Voltage (V)"),
  });

  if (curveCurrentChart) curveCurrentChart.destroy();
  curveCurrentChart = new Chart(document.getElementById("curve-current-chart").getContext("2d"), {
    type: "line",
    data: {
      labels,
      datasets: [
        { label: "Discharge current (A)", data: curve.dchg_current, borderColor: CHART_COLORS[3], pointRadius: 0 },
        { label: "Charge current (A)", data: curve.chg_current, borderColor: CHART_COLORS[0], pointRadius: 0 },
      ],
    },
    options: commonOpts("Current (A)"),
  });

  if (curveTempChart) curveTempChart.destroy();
  curveTempChart = new Chart(document.getElementById("curve-temp-chart").getContext("2d"), {
    type: "line",
    data: {
      labels,
      datasets: [
        { label: "Discharge temperature (°C)", data: curve.dchg_temperature, borderColor: CHART_COLORS[3], pointRadius: 0 },
        { label: "Charge temperature (°C)", data: curve.chg_temperature, borderColor: CHART_COLORS[0], pointRadius: 0 },
      ],
    },
    options: commonOpts("Temperature (°C)"),
  });
}

// --- Cross-cell comparison ---

function renderCompareChart() {
  const ctx = document.getElementById("compare-chart").getContext("2d");
  if (compareChart) compareChart.destroy();
  const groupCounters = {};
  const datasets = catalog.map((cell) => {
    const group = cell.form_factor === "OXFORD" ? "OXFORD" : "VL";
    const idx = groupCounters[group] ?? 0;
    groupCounters[group] = idx + 1;
    return {
      label: cell.cell_id,
      data: cellDataCache[cell.cell_id]
        ? cellDataCache[cell.cell_id].cycles.map((c, i) => ({ x: c, y: cellDataCache[cell.cell_id].soh_pct[i] }))
        : [],
      borderColor: compareColorFor(cell, idx),
      backgroundColor: "transparent",
      pointRadius: 0,
      borderWidth: 2.25,
      tension: 0.1,
    };
  });
  compareChart = new Chart(ctx, {
    type: "line",
    data: { datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      parsing: false,
      scales: {
        x: { type: "linear", title: { display: true, text: "Cycle" }, ...gridOpts() },
        y: { title: { display: true, text: "SOH (%)" }, min: 0, max: 105, ...gridOpts() },
      },
      plugins: { legend: { position: "bottom", labels: { usePointStyle: true, boxWidth: 8, boxHeight: 8, font: { weight: "600" } } } },
    },
  });
}

async function loadAllCellsForComparison() {
  await Promise.all(catalog.map((c) => loadCellData(c.cell_id)));
  renderCompareChart();
}

// --- Model inference ---

const C_RATE_VALUES = { "0.05C": 0.05, "0.1C": 0.1, "0.2C": 0.2, "0.5C": 0.5 };
const TEMP_VALUES = { RT: 25.0 };

function buildStaticVector(cell) {
  const cRate = C_RATE_VALUES[cell.c_rate] ?? parseFloat(cell.c_rate);
  const temp = TEMP_VALUES[cell.temp_condition] ?? parseFloat(cell.temp_condition);
  const isVl2020 = cell.form_factor === "VL2020" ? 1.0 : 0.0;
  return [cRate, temp, isVl2020];
}

function normalize(arr, mean, std) {
  return arr.map((v, i) => (v - mean[i]) / std[i]);
}

async function runPrediction(cellMeta, cellData, refCycleIdx) {
  const window = modelMeta.window;
  const featureKeys = modelMeta.feature_keys;
  const startIdx = refCycleIdx - window + 1;
  if (startIdx < 0) throw new Error("Not enough history before this cycle for a full window");

  // Build (window, n_features) raw sequence
  const seqRaw = [];
  for (let t = startIdx; t <= refCycleIdx; t++) {
    const row = featureKeys.map((key) => cellData[key][t]);
    seqRaw.push(row);
  }

  const seqNorm = seqRaw.map((row) => normalize(row, modelMeta.seq_mean, modelMeta.seq_std));
  const staticRaw = buildStaticVector(cellMeta);
  const staticNorm = normalize(staticRaw, modelMeta.static_mean, modelMeta.static_std);
  const lastSoh = cellData.soh_pct[refCycleIdx];

  const seqFlat = new Float32Array(seqNorm.flat());
  const seqTensor = new ort.Tensor("float32", seqFlat, [1, window, featureKeys.length]);
  const staticTensor = new ort.Tensor("float32", new Float32Array(staticNorm), [1, staticRaw.length]);
  const lastSohTensor = new ort.Tensor("float32", new Float32Array([lastSoh]), [1]);

  const output = await ortSession.run({
    x_seq: seqTensor,
    x_static: staticTensor,
    raw_last_soh: lastSohTensor,
  });

  const predKey = Object.keys(output)[0];
  return output[predKey].data[0];
}

function setupPredictPanel() {
  const slider = document.getElementById("cycle-slider");
  const label = document.getElementById("cycle-label");
  slider.addEventListener("input", () => {
    label.textContent = slider.value;
  });

  document.getElementById("predict-btn").addEventListener("click", async () => {
    if (!selectedCellId) return;
    const cellMeta = catalog.find((c) => c.cell_id === selectedCellId);
    const cellData = cellDataCache[selectedCellId];
    const refCycle = parseInt(slider.value, 10);
    const refIdx = cellData.cycles.indexOf(refCycle);
    const targetIdx = refIdx + 1;

    if (refIdx < modelMeta.window - 1) {
      document.getElementById("predict-result").innerHTML =
        `<p class="hint">Need at least ${modelMeta.window} cycles of history before this point.</p>`;
      return;
    }
    if (targetIdx >= cellData.cycles.length) {
      document.getElementById("predict-result").innerHTML =
        `<p class="hint">No next cycle available after the last tested cycle.</p>`;
      return;
    }

    const btn = document.getElementById("predict-btn");
    btn.disabled = true;
    btn.innerHTML = `${icon("clock")} Running...`;
    try {
      const pred = await runPrediction(cellMeta, cellData, refIdx);
      const actual = cellData.soh_pct[targetIdx];
      const err = Math.abs(pred - actual);
      const errClass = err < 1.5 ? "good" : "warn";
      document.getElementById("predict-result").innerHTML =
        statTile("brain", `Predicted SOH @ cycle ${cellData.cycles[targetIdx]}`, `${pred.toFixed(2)}%`, "accent") +
        statTile("battery", "Actual measured SOH", `${actual.toFixed(2)}%`) +
        statTile("scale", "Absolute error", `${err.toFixed(2)} pts`, errClass, errClass);
    } catch (e) {
      document.getElementById("predict-result").innerHTML = `<p class="hint">Error: ${e.message}</p>`;
    } finally {
      btn.disabled = false;
      btn.innerHTML = `${icon("brain")} Run prediction`;
    }
  });
}

async function init() {
  await loadCatalog();
  await loadLocoResults();
  await loadModelIndex();
  renderCellGrid();
  renderModelPicker();
  renderFleetStats();
  renderArchTable();
  renderLocoTable();
  updateAboutModelText();
  setupPredictPanel();
  await loadModel();
  await loadAllCellsForComparison();
  if (catalog.length) await selectCell(catalog[0].cell_id);
}

init().catch((e) => {
  console.error(e);
  document.body.insertAdjacentHTML(
    "afterbegin",
    `<div style="background:#f87171;color:#1a0000;padding:1rem;text-align:center;">Failed to initialize app: ${e.message}</div>`
  );
});
