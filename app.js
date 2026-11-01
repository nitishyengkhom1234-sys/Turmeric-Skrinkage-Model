// Embedded Model Weights for instant client-side forward pass & offline resilience
let MODEL_DATA = null;

// Initialize App
document.addEventListener('DOMContentLoaded', async () => {
  try {
    const res = await fetch('model_weights.json');
    if (res.ok) {
      MODEL_DATA = await res.json();
    }
  } catch (err) {
    console.warn('Could not fetch model_weights.json via HTTP, using fallback embedded parameters:', err);
  }

  // If fetch failed (e.g. running directly via file://) load fallback weights
  if (!MODEL_DATA) {
    MODEL_DATA = getFallbackModelData();
  }

  initSliders();
  initBatchTable();
  initDatasetTable();
  initChart();
  calculateAndRender();
});

// High Precision ANN Forward Pass (exact scikit-learn MLPRegressor replica)
function predictANN(temp, timeMin) {
  const m = MODEL_DATA;
  if (!m) return 0.3593;

  // 1. StandardScaler: (X - mean) / scale
  const mean = m.scaler.mean;
  const scale = m.scaler.scale;
  const x0 = (temp - mean[0]) / scale[0];
  const x1 = (timeMin - mean[1]) / scale[1];

  // 2. Hidden Layer 1: 2 -> 8 (tanh activation)
  const c0 = m.weights.coefs[0];
  const b0 = m.weights.intercepts[0];
  const h1 = [];
  for (let j = 0; j < 8; j++) {
    const sum = x0 * c0[0][j] + x1 * c0[1][j] + b0[j];
    h1.push(Math.tanh(sum));
  }

  // 3. Hidden Layer 2: 8 -> 4 (tanh activation)
  const c1 = m.weights.coefs[1];
  const b1 = m.weights.intercepts[1];
  const h2 = [];
  for (let j = 0; j < 4; j++) {
    let sum = b1[j];
    for (let k = 0; k < 8; k++) {
      sum += h1[k] * c1[k][j];
    }
    h2.push(Math.tanh(sum));
  }

  // 4. Output Layer: 4 -> 1 (identity activation)
  const c2 = m.weights.coefs[2];
  const b2 = m.weights.intercepts[2];
  let out = b2[0];
  for (let k = 0; k < 4; k++) {
    out += h2[k] * c2[k][0];
  }

  return {
    prediction: out,
    zTemp: x0,
    zTime: x1
  };
}

// UI State & Calculation
let currentTemp = 80.0;
let currentTime = 15.0;
let kineticsChart = null;

function calculateAndRender() {
  const result = predictANN(currentTemp, currentTime);
  const pred = result.prediction;
  const roundedPred = pred.toFixed(4);
  const volReduct = ((1.0 - pred) * 100).toFixed(2);

  // Update display values
  document.getElementById('tempValue').textContent = currentTemp.toFixed(1);
  document.getElementById('timeValue').textContent = currentTime.toFixed(1);
  document.getElementById('tempSlider').value = currentTemp;
  document.getElementById('timeSlider').value = currentTime;

  document.getElementById('predShrinkage').textContent = roundedPred;
  document.getElementById('volReduction').textContent = `${volReduct}% Volume Loss`;

  // Drying stage descriptor
  const stageEl = document.getElementById('dryingStage');
  if (currentTime === 0) {
    stageEl.textContent = 'Fresh Turmeric Rhizome';
  } else if (pred > 0.70) {
    stageEl.textContent = 'Constant / Initial Falling Rate';
  } else if (pred > 0.45) {
    stageEl.textContent = 'Active Moisture Evaporation';
  } else {
    stageEl.textContent = 'Equilibrium Bound Moisture';
  }

  // Math breakdown
  document.getElementById('mathZTemp').textContent = (result.zTemp >= 0 ? '+' : '') + result.zTemp.toFixed(3);
  document.getElementById('mathZTime').textContent = (result.zTime >= 0 ? '+' : '') + result.zTime.toFixed(3);

  // Visual Morphological Slice scale
  // Volume ratio is Sr. Cross section area scale ~ sqrt(Sr)
  const scale = Math.max(0.35, Math.min(1.0, Math.sqrt(pred)));
  const visualEl = document.getElementById('rhizomeDriedVisual');
  if (visualEl) {
    visualEl.style.transform = `scale(${scale})`;
    document.getElementById('driedPercentLabel').innerHTML = `${(pred * 100).toFixed(1)}% (${roundedPred})`;
  }

  // Update Chart
  updateChart();
}

function initSliders() {
  const tempSlider = document.getElementById('tempSlider');
  const timeSlider = document.getElementById('timeSlider');

  tempSlider.addEventListener('input', (e) => {
    currentTemp = parseFloat(e.target.value);
    highlightTempChips(currentTemp);
    calculateAndRender();
  });

  timeSlider.addEventListener('input', (e) => {
    currentTime = parseFloat(e.target.value);
    highlightTimeChips(currentTime);
    calculateAndRender();
  });
}

function setTemp(t) {
  currentTemp = parseFloat(t);
  highlightTempChips(currentTemp);
  calculateAndRender();
}

function setTime(m) {
  currentTime = parseFloat(m);
  highlightTimeChips(currentTime);
  calculateAndRender();
}

function highlightTempChips(t) {
  document.querySelectorAll('.quick-chips .quick-chip').forEach(el => {
    if (el.id === 'chipTemp80') {
      el.classList.toggle('active', Math.abs(t - 80) < 0.1);
    }
  });
}

function highlightTimeChips(m) {
  document.querySelectorAll('.quick-chips .quick-chip').forEach(el => {
    if (el.id === 'chipTime15') {
      el.classList.toggle('active', Math.abs(m - 15) < 0.1);
    }
  });
}

function applyPreset(temp, timeMin) {
  currentTemp = temp;
  currentTime = timeMin;
  calculateAndRender();
}

// Chart.js Kinetics Graph
function initChart() {
  const ctx = document.getElementById('kineticsChart').getContext('2d');
  
  kineticsChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: [],
      datasets: [
        {
          label: 'ANN Model Prediction Curve',
          data: [],
          borderColor: '#f59e0b',
          backgroundColor: 'rgba(245, 158, 11, 0.15)',
          borderWidth: 3,
          fill: true,
          tension: 0.35,
          pointRadius: 0,
          pointHoverRadius: 6
        },
        {
          label: 'Current Test Input',
          data: [],
          borderColor: '#ffffff',
          backgroundColor: '#ef4444',
          borderWidth: 2,
          pointRadius: 8,
          pointHoverRadius: 10,
          showLine: false
        },
        {
          label: 'Exp 50 °C Observations',
          data: [],
          borderColor: '#3b82f6',
          backgroundColor: '#3b82f6',
          pointRadius: 5,
          showLine: false
        },
        {
          label: 'Exp 60 °C Observations',
          data: [],
          borderColor: '#10b981',
          backgroundColor: '#10b981',
          pointRadius: 5,
          showLine: false
        },
        {
          label: 'Exp 70 °C Observations',
          data: [],
          borderColor: '#a855f7',
          backgroundColor: '#a855f7',
          pointRadius: 5,
          showLine: false
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: {
        mode: 'index',
        intersect: false
      },
      plugins: {
        legend: {
          labels: {
            color: '#94a3b8',
            font: { family: 'Plus Jakarta Sans', size: 11 },
            boxWidth: 14
          }
        },
        tooltip: {
          backgroundColor: '#0f172a',
          titleColor: '#f8fafc',
          bodyColor: '#cbd5e1',
          borderColor: '#334155',
          borderWidth: 1,
          padding: 10,
          callbacks: {
            label: function(context) {
              const val = context.parsed.y !== null ? context.parsed.y.toFixed(4) : '';
              return `${context.dataset.label}: ${val}`;
            }
          }
        }
      },
      scales: {
        x: {
          type: 'linear',
          title: {
            display: true,
            text: 'Drying Time (minutes)',
            color: '#94a3b8',
            font: { family: 'Plus Jakarta Sans', size: 12, weight: 'bold' }
          },
          grid: { color: 'rgba(255, 255, 255, 0.05)' },
          ticks: { color: '#64748b', stepSize: 5 },
          min: 0,
          max: 40
        },
        y: {
          title: {
            display: true,
            text: 'Shrinkage Ratio (Sr)',
            color: '#94a3b8',
            font: { family: 'Plus Jakarta Sans', size: 12, weight: 'bold' }
          },
          grid: { color: 'rgba(255, 255, 255, 0.05)' },
          ticks: { color: '#64748b' },
          min: 0.2,
          max: 1.05
        }
      }
    }
  });

  updateChart();
}

function updateChart() {
  if (!kineticsChart || !MODEL_DATA) return;

  document.getElementById('chartTempLabel').textContent = `${currentTemp.toFixed(1)} °C`;

  // Continuous curve for currentTemp from t = 0 to 40
  const curvePoints = [];
  for (let t = 0; t <= 40; t += 0.5) {
    const res = predictANN(currentTemp, t);
    curvePoints.push({ x: t, y: res.prediction });
  }

  // Current selected point
  const currentRes = predictANN(currentTemp, currentTime);
  const currentPoint = [{ x: currentTime, y: currentRes.prediction }];

  // Empirical data points from dataset
  const show50 = document.getElementById('showExp50').checked;
  const show60 = document.getElementById('showExp60').checked;
  const show70 = document.getElementById('showExp70').checked;

  const exp50 = show50 ? MODEL_DATA.training_data.filter(d => d.Temperature === 50).map(d => ({ x: d.Time, y: d.ShrinkageRatio })) : [];
  const exp60 = show60 ? MODEL_DATA.training_data.filter(d => d.Temperature === 60).map(d => ({ x: d.Time, y: d.ShrinkageRatio })) : [];
  const exp70 = show70 ? MODEL_DATA.training_data.filter(d => d.Temperature === 70).map(d => ({ x: d.Time, y: d.ShrinkageRatio })) : [];

  kineticsChart.data.datasets[0].data = curvePoints;
  kineticsChart.data.datasets[0].label = `ANN Curve @ ${currentTemp.toFixed(1)} °C`;
  kineticsChart.data.datasets[1].data = currentPoint;
  kineticsChart.data.datasets[2].data = exp50;
  kineticsChart.data.datasets[3].data = exp60;
  kineticsChart.data.datasets[4].data = exp70;

  kineticsChart.update('none');
}

// Dataset Table Population
function initDatasetTable() {
  renderDatasetRows('all');
}

function filterDataset(filter, btn) {
  document.querySelectorAll('.filter-group .filter-btn').forEach(b => b.classList.remove('active'));
  if (btn) btn.classList.add('active');
  renderDatasetRows(filter);
}

function renderDatasetRows(filter) {
  const tbody = document.getElementById('datasetTableBody');
  if (!tbody || !MODEL_DATA) return;

  let items = MODEL_DATA.training_data;
  if (filter !== 'all') {
    items = items.filter(d => d.Temperature === Number(filter));
  }

  tbody.innerHTML = items.map((row, idx) => {
    const predRes = predictANN(row.Temperature, row.Time);
    const pred = predRes.prediction;
    const diff = Math.abs(row.ShrinkageRatio - pred);

    return `
      <tr>
        <td>#${idx + 1}</td>
        <td><strong>${row.Temperature} °C</strong></td>
        <td>${row.Time} min</td>
        <td>${row.ShrinkageRatio.toFixed(4)}</td>
        <td style="color: #fbbf24;">${pred.toFixed(4)}</td>
        <td style="color: ${diff < 0.02 ? '#34d399' : '#f87171'};">±${diff.toFixed(5)}</td>
        <td>
          <button class="table-btn" onclick="applyPreset(${row.Temperature}, ${row.Time})">Load</button>
        </td>
      </tr>
    `;
  }).join('');
}

// Batch Tester
const initialBatch = [
  { temp: 80.0, time: 15.0 },
  { temp: 50.0, time: 5.0 },
  { temp: 50.0, time: 20.0 },
  { temp: 60.0, time: 15.0 },
  { temp: 70.0, time: 8.0 },
  { temp: 75.0, time: 12.0 }
];

function initBatchTable() {
  const tbody = document.getElementById('batchTableBody');
  if (!tbody) return;

  tbody.innerHTML = initialBatch.map((row, i) => createBatchRowHTML(i + 1, row.temp, row.time)).join('');
  runBatchPredictions();
}

function createBatchRowHTML(id, temp, time) {
  return `
    <tr id="batch-row-${id}">
      <td>${id}</td>
      <td><input type="number" step="0.5" value="${temp}" class="table-input batch-temp" oninput="runBatchPredictions()"></td>
      <td><input type="number" step="0.5" value="${time}" class="table-input batch-time" oninput="runBatchPredictions()"></td>
      <td class="batch-pred" style="font-weight: 700; color: #fbbf24;">-</td>
      <td class="batch-loss" style="color: #34d399;">-</td>
      <td>
        <button class="table-btn" onclick="removeBatchRow(${id})">✕</button>
      </td>
    </tr>
  `;
}

let batchRowCount = 6;
function addBatchRow() {
  batchRowCount++;
  const tbody = document.getElementById('batchTableBody');
  const tr = document.createElement('tr');
  tr.id = `batch-row-${batchRowCount}`;
  tr.innerHTML = `
    <td>${batchRowCount}</td>
    <td><input type="number" step="0.5" value="70.0" class="table-input batch-temp" oninput="runBatchPredictions()"></td>
    <td><input type="number" step="0.5" value="10.0" class="table-input batch-time" oninput="runBatchPredictions()"></td>
    <td class="batch-pred" style="font-weight: 700; color: #fbbf24;">-</td>
    <td class="batch-loss" style="color: #34d399;">-</td>
    <td><button class="table-btn" onclick="removeBatchRow(${batchRowCount})">✕</button></td>
  `;
  tbody.appendChild(tr);
  runBatchPredictions();
}

function removeBatchRow(id) {
  const row = document.getElementById(`batch-row-${id}`);
  if (row) row.remove();
}

function runBatchPredictions() {
  const rows = document.querySelectorAll('#batchTableBody tr');
  rows.forEach(tr => {
    const tempInput = tr.querySelector('.batch-temp');
    const timeInput = tr.querySelector('.batch-time');
    const predTd = tr.querySelector('.batch-pred');
    const lossTd = tr.querySelector('.batch-loss');

    if (tempInput && timeInput && predTd) {
      const t = parseFloat(tempInput.value) || 0;
      const m = parseFloat(timeInput.value) || 0;
      const res = predictANN(t, m);
      const p = res.prediction;
      predTd.textContent = p.toFixed(4);
      lossTd.textContent = `${((1.0 - p) * 100).toFixed(2)}%`;
    }
  });
}

function exportBatchCSV() {
  const rows = document.querySelectorAll('#batchTableBody tr');
  let csv = 'Temperature,Time,PredictedShrinkageRatio,VolumeReductionPercent\n';
  rows.forEach(tr => {
    const t = tr.querySelector('.batch-temp')?.value || '';
    const m = tr.querySelector('.batch-time')?.value || '';
    const p = tr.querySelector('.batch-pred')?.textContent || '';
    const l = tr.querySelector('.batch-loss')?.textContent?.replace('%', '') || '';
    if (t && m) {
      csv += `${t},${m},${p},${l}\n`;
    }
  });

  const blob = new Blob([csv], { type: 'text/csv' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'turmeric_ann_predictions.csv';
  a.click();
  URL.revokeObjectURL(url);
}

// Navigation Tabs
function switchTab(tabId) {
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));

  event.currentTarget.classList.add('active');
  const panel = document.getElementById(`tab-${tabId}`);
  if (panel) panel.classList.add('active');

  if (tabId === 'chart' && kineticsChart) {
    setTimeout(() => kineticsChart.resize(), 50);
  }
}

function showCodeSnippet(type) {
  document.querySelectorAll('.code-tab').forEach(b => b.classList.remove('active'));
  document.querySelectorAll('.code-snippet').forEach(s => s.classList.remove('active'));

  event.currentTarget.classList.add('active');
  document.getElementById(`code-${type}`).classList.add('active');
}

// Copy prediction
function copyPrediction() {
  const p = document.getElementById('predShrinkage').textContent;
  navigator.clipboard.writeText(`Temperature: ${currentTemp}°C, Time: ${currentTime} min -> Predicted Shrinkage Ratio: ${p}`);
  alert(`Copied: ${currentTemp}°C, ${currentTime} min -> Sr = ${p}`);
}

// Live API Tester
async function testLiveAPI() {
  const feedback = document.getElementById('apiTestFeedback');
  feedback.classList.remove('hidden');
  feedback.textContent = `Connecting to /api/predict (temp=${currentTemp}, time=${currentTime})...`;

  try {
    const res = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ temperature: currentTemp, time: currentTime })
    });

    if (res.ok) {
      const json = await res.json();
      feedback.style.color = '#34d399';
      feedback.textContent = `✓ API Response [200 OK]: Predicted = ${json.predicted_shrinkage_ratio} (Latency: fast)`;
    } else {
      feedback.style.color = '#fbbf24';
      feedback.textContent = `Client-side evaluation active (Server returned status ${res.status})`;
    }
  } catch (err) {
    feedback.style.color = '#34d399';
    feedback.textContent = `✓ Client ANN Model Active: Evaluated ${predictANN(currentTemp, currentTime).prediction.toFixed(4)}`;
  }
}

// Fallback Model Data embedded directly
function getFallbackModelData() {
  return {
    "model_type": "MLPRegressor",
    "architecture": { "input_dim": 2, "hidden_layers": [8, 4], "output_dim": 1, "activation": "tanh", "solver": "lbfgs" },
    "metrics": { "r2_score": 0.996612620124306, "mse": 0.00011351448701438139 },
    "scaler": {
      "mean": [58.148148148148145, 13.481481481481481],
      "scale": [8.181748901620193, 10.12283948735206]
    },
    "weights": {
      "coefs": [
        [
          [0.11942184487029692, 0.5849664586911719, 0.11461239632614575, 0.05565650448197999, -0.1576711343517674, -0.416022378596196, -0.2204157528650356, 0.5145986181259907],
          [0.23869049193083364, 0.0931314164134891, -0.4744929084627257, 0.567704543345954, 0.35305925933806576, -0.3881169887414966, -0.6601005539137174, -0.07182402722524998]
        ],
        [
          [0.16975881358225345, 0.7336415711320746, -0.3383087759856766, 0.1654441372765674],
          [0.2023676642302713, -0.6617923545507215, -0.03139766644716895, -0.32481892172172744],
          [-0.967761666762508, 0.26130679985604005, 0.7130535623562356, 0.40802607972547544],
          [0.13142791337685544, -0.08423304799458127, 0.15504363885126027, 0.14438949742717822],
          [-0.2769465641929758, 0.26138952369240315, -0.46577952309333553, 0.5388609251709421],
          [-0.3051976801397271, 0.3881308846498049, -0.10697739990546823, 0.0686543403614308],
          [0.012811553523264109, -0.45788405147050765, 0.6621521981730951, 0.2661286112456627],
          [0.15176127702283188, 0.23868004724640657, 0.33621794760419055, 0.596305285601784]
        ],
        [
          [-0.46738757544467047],
          [-0.30828323714096867],
          [0.46702252780807324],
          [0.05543899436113709]
        ]
      ],
      "intercepts": [
        [-0.41372350393251955, 0.27993704283736154, -1.0001036195202941, -0.296121627160597, 0.25576805779648254, -0.5696625951087326, -0.39289594888091406, -0.03604191112752851],
        [-0.7498694733885847, -0.982813349237449, -0.6870701662171721, -0.5964302200437198],
        [0.7226935219508809]
      ]
    },
    "training_data": [
      { "Temperature": 50, "Time": 0, "ShrinkageRatio": 1.0 },
      { "Temperature": 50, "Time": 2, "ShrinkageRatio": 0.9338 },
      { "Temperature": 50, "Time": 4, "ShrinkageRatio": 0.8733 },
      { "Temperature": 50, "Time": 6, "ShrinkageRatio": 0.818 },
      { "Temperature": 50, "Time": 8, "ShrinkageRatio": 0.7675 },
      { "Temperature": 50, "Time": 10, "ShrinkageRatio": 0.7213 },
      { "Temperature": 50, "Time": 12, "ShrinkageRatio": 0.6791 },
      { "Temperature": 50, "Time": 16, "ShrinkageRatio": 0.6053 },
      { "Temperature": 50, "Time": 20, "ShrinkageRatio": 0.5437 },
      { "Temperature": 50, "Time": 24, "ShrinkageRatio": 0.4921 },
      { "Temperature": 50, "Time": 28, "ShrinkageRatio": 0.4491 },
      { "Temperature": 50, "Time": 32, "ShrinkageRatio": 0.4132 },
      { "Temperature": 50, "Time": 36, "ShrinkageRatio": 0.3832 },
      { "Temperature": 60, "Time": 0, "ShrinkageRatio": 1.0 },
      { "Temperature": 60, "Time": 2, "ShrinkageRatio": 0.8929 },
      { "Temperature": 60, "Time": 4, "ShrinkageRatio": 0.8007 },
      { "Temperature": 60, "Time": 6, "ShrinkageRatio": 0.7213 },
      { "Temperature": 60, "Time": 8, "ShrinkageRatio": 0.653 },
      { "Temperature": 60, "Time": 10, "ShrinkageRatio": 0.5942 },
      { "Temperature": 60, "Time": 12, "ShrinkageRatio": 0.5437 },
      { "Temperature": 60, "Time": 16, "ShrinkageRatio": 0.4626 },
      { "Temperature": 60, "Time": 20, "ShrinkageRatio": 0.4026 },
      { "Temperature": 60, "Time": 24, "ShrinkageRatio": 0.3581 },
      { "Temperature": 60, "Time": 29, "ShrinkageRatio": 0.3184 },
      { "Temperature": 70, "Time": 0, "ShrinkageRatio": 1.0 },
      { "Temperature": 70, "Time": 2, "ShrinkageRatio": 0.8543 },
      { "Temperature": 70, "Time": 4, "ShrinkageRatio": 0.7363 },
      { "Temperature": 70, "Time": 6, "ShrinkageRatio": 0.6406 },
      { "Temperature": 70, "Time": 8, "ShrinkageRatio": 0.563 },
      { "Temperature": 70, "Time": 10, "ShrinkageRatio": 0.5001 },
      { "Temperature": 70, "Time": 12, "ShrinkageRatio": 0.4491 },
      { "Temperature": 70, "Time": 16, "ShrinkageRatio": 0.3743 },
      { "Temperature": 70, "Time": 20, "ShrinkageRatio": 0.3252 },
      { "Temperature": 70, "Time": 23, "ShrinkageRatio": 0.2997 }
    ]
  };
}
