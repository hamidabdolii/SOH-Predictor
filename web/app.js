// VL Coin-Cell SOH Predictor — client-side app.
// Loads pre-computed per-cell JSON (data/<cell_id>.json), runs the ONNX LSTM
// in-browser for next-cycle SOH prediction, and renders comparison charts.

const DATA_BASE = "data";
const MODEL_BASE = "model";

let catalog = [];
let cellDataCache = {}; // cell_id -> parsed JSON
let modelMeta = null;
let ortSession = null;
let selectedCellId = null;

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

async function loadCellData(cellId) {
  if (cellDataCache[cellId]) return cellDataCache[cellId];
  const data = await fetchJSON(`${DATA_BASE}/${cellId}.json`);
  cellDataCache[cellId] = data;
  return data;
}

async function loadModel() {
  modelMeta = await fetchJSON(`${MODEL_BASE}/soh_lstm_meta.json`);
  // Single-threaded WASM: GitHub Pages doesn't set the COOP/COEP headers
  // required for SharedArrayBuffer, so multi-threaded WASM execution isn't
  // available there. The model is tiny (16 hidden units), so single-thread
  // CPU inference is effectively instant regardless.
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.wasmPaths = "vendor/";
  ortSession = await ort.InferenceSession.create(`${MODEL_BASE}/soh_lstm.onnx`);
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

function renderCellGrid() {
  const grid = document.getElementById("cell-grid");
  grid.innerHTML = "";
  catalog.forEach((cell) => {
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
    grid.appendChild(card);
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

  if (!data.loco) {
    btn.disabled = true;
    btn.innerHTML = `${icon("target")} Not available for this cell`;
    statsRow.innerHTML = `<p class="hint">This cell wasn't part of the leave-one-cell-out validation set, so no genuine out-of-sample predictions are available for it yet.</p>`;
    return;
  }

  btn.disabled = false;
  btn.innerHTML = `${icon("play")} Run full prediction`;
  btn.onclick = () => runTrajectoryAnimation(data);
}

async function runTrajectoryAnimation(data) {
  const myToken = ++trajectoryRunToken;
  const btn = document.getElementById("run-trajectory-btn");
  const statsRow = document.getElementById("trajectory-stats");
  const n = data.loco.target_cycle.length;

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

function updateTrajectoryStats(data, shownCount) {
  const trueVals = data.loco.true_soh_pct.slice(0, shownCount);
  const predVals = data.loco.predicted_soh_pct.slice(0, shownCount);
  const errors = trueVals.map((t, i) => Math.abs(t - predVals[i]));
  const mae = errors.reduce((a, b) => a + b, 0) / errors.length;
  const maxErr = Math.max(...errors);

  const statsRow = document.getElementById("trajectory-stats");
  const maeGood = mae < 1.5;
  statsRow.innerHTML =
    statTile("target", "Predictions made", `${shownCount} / ${data.loco.target_cycle.length}`, "accent") +
    statTile("scale", "Mean absolute error", `${mae.toFixed(2)} pts`, maeGood ? "good" : "warn", maeGood ? "good" : "warn") +
    statTile("trend", "Max error so far", `${maxErr.toFixed(2)} pts`);
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

  if (data.loco) {
    const trueShown = data.loco.target_cycle.slice(0, shownCount).map((c, i) => ({ x: c, y: data.loco.true_soh_pct[i] }));
    const predShown = data.loco.target_cycle.slice(0, shownCount).map((c, i) => ({ x: c, y: data.loco.predicted_soh_pct[i] }));
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
        label: "Predicted SOH (leave-one-cell-out)",
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
  const datasets = catalog.map((cell, i) => ({
    label: cell.cell_id,
    data: cellDataCache[cell.cell_id]
      ? cellDataCache[cell.cell_id].cycles.map((c, idx) => ({ x: c, y: cellDataCache[cell.cell_id].soh_pct[idx] }))
      : [],
    borderColor: CHART_COLORS[i % CHART_COLORS.length],
    backgroundColor: "transparent",
    pointRadius: 0,
    borderWidth: 2.25,
    tension: 0.1,
  }));
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

const C_RATE_VALUES = { "0.05C": 0.05, "0.1C": 0.1, "0.2C": 0.2 };
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
  renderCellGrid();
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
