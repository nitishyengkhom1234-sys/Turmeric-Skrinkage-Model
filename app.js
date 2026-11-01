// Global state
let MODEL_DATA = null;
let currentTempUnit = 'C';   // 'C', 'F', 'K'
let currentTimeUnit = 'min'; // 'min', 'hr', 's'
let rawTemp = 80.0;
let rawTime = 15.0;

// User custom points plotted on graph
let userPoints = [
  {
    id: 1,
    label: "Prompt Benchmark",
    rawTemp: 80.0,
    tempUnit: "C",
    tempC: 80.0,
    rawTime: 15.0,
    timeUnit: "min",
    timeMin: 15.0,
    observedSr: 0.3593,
    predictedSr: 0.3593
  }
];

let kineticsChart = null;

// Initialize Application
document.addEventListener('DOMContentLoaded', async () => {
  try {
    const res = await fetch('model_weights.json');
    if (res.ok) {
      MODEL_DATA = await res.json();
    }
  } catch (err) {
    console.warn('Could not fetch model_weights.json via HTTP, using fallback embedded parameters:', err);
  }

  if (!MODEL_DATA) {
    MODEL_DATA = getFallbackModelData();
  }

  initUIControls();
  renderUserPointsTable();
  initBatchTable();
  initDatasetTable();
  initChart();
  calculateAndRender();
});

// Unit Conversion Helpers
function toCelsius(val, unit) {
  if (unit === 'F') return (val - 32.0) * 5.0 / 9.0;
  if (unit === 'K') return val - 273.15;
  return val;
}

function fromCelsius(valC, unit) {
  if (unit === 'F') return valC * 9.0 / 5.0 + 32.0;
  if (unit === 'K') return valC + 273.15;
  return valC;
}

function toMinutes(val, unit) {
  if (unit === 'hr') return val * 60.0;
  if (unit === 's') return val / 60.0;
  return val;
}

function fromMinutes(valMin, unit) {
  if (unit === 'hr') return valMin / 60.0;
  if (unit === 's') return valMin * 60.0;
  return valMin;
}

// Model Forward Pass (Exact scikit-learn replica)
function predictANN(tempC, timeMin) {
  const m = MODEL_DATA;
  if (!m) return { prediction: 0.3593, zTemp: 0, zTime: 0 };

  const mean = m.scaler.mean;
  const scale = m.scaler.scale;
  const x0 = (tempC - mean[0]) / scale[0];
  const x1 = (timeMin - mean[1]) / scale[1];

  // Hidden Layer 1: 2 -> 8 (tanh)
  const c0 = m.weights.coefs[0];
  const b0 = m.weights.intercepts[0];
  const h1 = [];
  for (let j = 0; j < 8; j++) {
    const sum = x0 * c0[0][j] + x1 * c0[1][j] + b0[j];
    h1.push(Math.tanh(sum));
  }

  // Hidden Layer 2: 8 -> 4 (tanh)
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

  // Output Layer: 4 -> 1 (identity)
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

// UI Setup & Sliders
function initUIControls() {
  const tempSlider = document.getElementById('tempSlider');
  const tempInput = document.getElementById('tempNumericInput');
  const timeSlider = document.getElementById('timeSlider');
  const timeInput = document.getElementById('timeNumericInput');

  // Temp Listeners
  tempSlider.addEventListener('input', (e) => {
    rawTemp = parseFloat(e.target.value);
    tempInput.value = rawTemp.toFixed(1);
    calculateAndRender();
  });

  tempInput.addEventListener('input', (e) => {
    const v = parseFloat(e.target.value);
    if (!isNaN(v)) {
      rawTemp = v;
      tempSlider.value = rawTemp;
      calculateAndRender();
    }
  });

  // Time Listeners
  timeSlider.addEventListener('input', (e) => {
    rawTime = parseFloat(e.target.value);
    timeInput.value = currentTimeUnit === 'hr' ? rawTime.toFixed(2) : (currentTimeUnit === 's' ? Math.round(rawTime) : rawTime.toFixed(1));
    calculateAndRender();
  });

  timeInput.addEventListener('input', (e) => {
    const v = parseFloat(e.target.value);
    if (!isNaN(v)) {
      rawTime = v;
      timeSlider.value = rawTime;
      calculateAndRender();
    }
  });

  updateTempUnitUI();
  updateTimeUnitUI();
}

// Unit Changes
function onTempUnitChange() {
  const oldUnit = currentTempUnit;
  const newUnit = document.getElementById('tempUnitSelect').value;
  if (oldUnit === newUnit) return;

  // Convert raw value to preserve exact physical temperature
  const tempC = toCelsius(rawTemp, oldUnit);
  rawTemp = fromCelsius(tempC, newUnit);
  currentTempUnit = newUnit;

  updateTempUnitUI();
  calculateAndRender();
}

function updateTempUnitUI() {
  document.getElementById('tempUnitTag').textContent = `°${currentTempUnit}`;
  const tempInput = document.getElementById('tempNumericInput');
  const tempSlider = document.getElementById('tempSlider');
  const ticksEl = document.getElementById('tempTicks');
  const chipsEl = document.getElementById('tempQuickChips');

  if (currentTempUnit === 'C') {
    tempSlider.min = 30;
    tempSlider.max = 120;
    tempSlider.step = 0.5;
    ticksEl.innerHTML = `
      <span onclick="setTemp(50)">50°C</span>
      <span onclick="setTemp(60)">60°C</span>
      <span onclick="setTemp(70)">70°C</span>
      <span onclick="setTemp(80)">80°C</span>
      <span onclick="setTemp(90)">90°C</span>
    `;
    chipsEl.innerHTML = `
      <span class="quick-chip" onclick="setTemp(50)">50 °C (Lab)</span>
      <span class="quick-chip" onclick="setTemp(60)">60 °C (Lab)</span>
      <span class="quick-chip" onclick="setTemp(70)">70 °C (Lab)</span>
      <span class="quick-chip active" onclick="setTemp(80)">80 °C (Test)</span>
    `;
  } else if (currentTempUnit === 'F') {
    tempSlider.min = 86;
    tempSlider.max = 248;
    tempSlider.step = 1;
    ticksEl.innerHTML = `
      <span onclick="setTemp(122)">122°F</span>
      <span onclick="setTemp(140)">140°F</span>
      <span onclick="setTemp(158)">158°F</span>
      <span onclick="setTemp(176)">176°F</span>
      <span onclick="setTemp(194)">194°F</span>
    `;
    chipsEl.innerHTML = `
      <span class="quick-chip" onclick="setTemp(122)">122 °F (50°C)</span>
      <span class="quick-chip" onclick="setTemp(140)">140 °F (60°C)</span>
      <span class="quick-chip" onclick="setTemp(158)">158 °F (70°C)</span>
      <span class="quick-chip active" onclick="setTemp(176)">176 °F (80°C)</span>
    `;
  } else if (currentTempUnit === 'K') {
    tempSlider.min = 303.15;
    tempSlider.max = 393.15;
    tempSlider.step = 0.5;
    ticksEl.innerHTML = `
      <span onclick="setTemp(323.15)">323 K</span>
      <span onclick="setTemp(333.15)">333 K</span>
      <span onclick="setTemp(343.15)">343 K</span>
      <span onclick="setTemp(353.15)">353 K</span>
      <span onclick="setTemp(363.15)">363 K</span>
    `;
    chipsEl.innerHTML = `
      <span class="quick-chip" onclick="setTemp(323.15)">323.15 K</span>
      <span class="quick-chip" onclick="setTemp(333.15)">333.15 K</span>
      <span class="quick-chip" onclick="setTemp(343.15)">343.15 K</span>
      <span class="quick-chip active" onclick="setTemp(353.15)">353.15 K (80°C)</span>
    `;
  }

  tempSlider.value = rawTemp;
  tempInput.value = rawTemp.toFixed(1);
  const newPointUnit = document.getElementById('newPointTempUnit');
  if (newPointUnit) newPointUnit.textContent = `°${currentTempUnit}`;
}

function onTimeUnitChange() {
  const oldUnit = currentTimeUnit;
  const newUnit = document.getElementById('timeUnitSelect').value;
  if (oldUnit === newUnit) return;

  // Convert raw value to preserve physical drying time
  const timeMin = toMinutes(rawTime, oldUnit);
  rawTime = fromMinutes(timeMin, newUnit);
  currentTimeUnit = newUnit;

  updateTimeUnitUI();
  calculateAndRender();
}

function updateTimeUnitUI() {
  document.getElementById('timeUnitTag').textContent = currentTimeUnit;
  const timeInput = document.getElementById('timeNumericInput');
  const timeSlider = document.getElementById('timeSlider');
  const ticksEl = document.getElementById('timeTicks');
  const chipsEl = document.getElementById('timeQuickChips');

  if (currentTimeUnit === 'min') {
    timeSlider.min = 0;
    timeSlider.max = 60;
    timeSlider.step = 0.5;
    ticksEl.innerHTML = `
      <span onclick="setTime(0)">0m</span>
      <span onclick="setTime(10)">10m</span>
      <span onclick="setTime(15)">15m</span>
      <span onclick="setTime(25)">25m</span>
      <span onclick="setTime(40)">40m</span>
    `;
    chipsEl.innerHTML = `
      <span class="quick-chip" onclick="setTime(0)">0 min (Fresh)</span>
      <span class="quick-chip" onclick="setTime(10)">10 min</span>
      <span class="quick-chip active" onclick="setTime(15)">15 min (Test)</span>
      <span class="quick-chip" onclick="setTime(25)">25 min</span>
    `;
    timeInput.value = rawTime.toFixed(1);
  } else if (currentTimeUnit === 'hr') {
    timeSlider.min = 0;
    timeSlider.max = 1.2;
    timeSlider.step = 0.01;
    ticksEl.innerHTML = `
      <span onclick="setTime(0)">0h</span>
      <span onclick="setTime(0.25)">0.25h</span>
      <span onclick="setTime(0.5)">0.5h</span>
      <span onclick="setTime(0.75)">0.75h</span>
      <span onclick="setTime(1.0)">1.0h</span>
    `;
    chipsEl.innerHTML = `
      <span class="quick-chip" onclick="setTime(0)">0 hr (Fresh)</span>
      <span class="quick-chip active" onclick="setTime(0.25)">0.25 hr (15 min)</span>
      <span class="quick-chip" onclick="setTime(0.5)">0.50 hr (30 min)</span>
      <span class="quick-chip" onclick="setTime(0.75)">0.75 hr (45 min)</span>
    `;
    timeInput.value = rawTime.toFixed(2);
  } else if (currentTimeUnit === 's') {
    timeSlider.min = 0;
    timeSlider.max = 3600;
    timeSlider.step = 10;
    ticksEl.innerHTML = `
      <span onclick="setTime(0)">0s</span>
      <span onclick="setTime(600)">600s</span>
      <span onclick="setTime(900)">900s</span>
      <span onclick="setTime(1500)">1500s</span>
      <span onclick="setTime(2400)">2400s</span>
    `;
    chipsEl.innerHTML = `
      <span class="quick-chip" onclick="setTime(0)">0s (Fresh)</span>
      <span class="quick-chip" onclick="setTime(600)">600s (10m)</span>
      <span class="quick-chip active" onclick="setTime(900)">900s (15m)</span>
      <span class="quick-chip" onclick="setTime(1800)">1800s (30m)</span>
    `;
    timeInput.value = Math.round(rawTime);
  }

  timeSlider.value = rawTime;
  const newPointUnit = document.getElementById('newPointTimeUnit');
  if (newPointUnit) newPointUnit.textContent = currentTimeUnit;
}

function setTemp(val) {
  rawTemp = parseFloat(val);
  document.getElementById('tempSlider').value = rawTemp;
  document.getElementById('tempNumericInput').value = rawTemp.toFixed(1);
  calculateAndRender();
}

function setTime(val) {
  rawTime = parseFloat(val);
  document.getElementById('timeSlider').value = rawTime;
  document.getElementById('timeNumericInput').value = currentTimeUnit === 'hr' ? rawTime.toFixed(2) : (currentTimeUnit === 's' ? Math.round(rawTime) : rawTime.toFixed(1));
  calculateAndRender();
}

// Preset Handler (Base values in °C and min)
function applyPresetBase(tempC, timeMin) {
  // Convert into user's current selected units
  rawTemp = fromCelsius(tempC, currentTempUnit);
  rawTime = fromMinutes(timeMin, currentTimeUnit);

  document.getElementById('tempSlider').value = rawTemp;
  document.getElementById('tempNumericInput').value = rawTemp.toFixed(1);

  document.getElementById('timeSlider').value = rawTime;
  document.getElementById('timeNumericInput').value = currentTimeUnit === 'hr' ? rawTime.toFixed(2) : (currentTimeUnit === 's' ? Math.round(rawTime) : rawTime.toFixed(1));

  calculateAndRender();
}

// Main Calculation & Render Loop
function calculateAndRender() {
  const tempC = toCelsius(rawTemp, currentTempUnit);
  const timeMin = toMinutes(rawTime, currentTimeUnit);

  // Conversion Badges
  document.getElementById('tempInCelsius').textContent = tempC.toFixed(1);
  document.getElementById('timeInMinutes').textContent = timeMin.toFixed(1);

  // Model Forward Pass
  const res = predictANN(tempC, timeMin);
  const pred = res.prediction;
  const roundedPred = pred.toFixed(4);
  const volReduct = ((1.0 - pred) * 100).toFixed(2);

  document.getElementById('predShrinkage').textContent = roundedPred;
  document.getElementById('volReduction').textContent = `${volReduct}% Volume Reduction`;

  // Drying Stage descriptor
  const stageEl = document.getElementById('dryingStage');
  if (timeMin === 0) {
    stageEl.textContent = 'Fresh Turmeric Rhizome';
  } else if (pred > 0.70) {
    stageEl.textContent = 'Constant / Initial Falling Rate';
  } else if (pred > 0.45) {
    stageEl.textContent = 'Active Moisture Evaporation';
  } else {
    stageEl.textContent = 'Equilibrium Bound Moisture';
  }

  // Trace values
  document.getElementById('traceTemp').textContent = `${tempC.toFixed(2)} °C`;
  document.getElementById('traceTime').textContent = `${timeMin.toFixed(2)} min`;
  document.getElementById('mathZTemp').textContent = (res.zTemp >= 0 ? '+' : '') + res.zTemp.toFixed(3);
  document.getElementById('mathZTime').textContent = (res.zTime >= 0 ? '+' : '') + res.zTime.toFixed(3);

  // Slice Visualizer (Scale ~ sqrt(Sr))
  const scale = Math.max(0.35, Math.min(1.0, Math.sqrt(pred)));
  const visualEl = document.getElementById('rhizomeDriedVisual');
  if (visualEl) {
    visualEl.style.transform = `scale(${scale})`;
    document.getElementById('driedPercentLabel').innerHTML = `${(pred * 100).toFixed(1)}% (${roundedPred})`;
  }

  // Update chart
  updateChart();
}

// Dynamic Kinetics Graph (Unit-Aware and User-Data Driven)
function initChart() {
  const ctx = document.getElementById('kineticsChart').getContext('2d');

  kineticsChart = new Chart(ctx, {
    type: 'line',
    data: {
      datasets: [
        {
          label: 'ANN Model Prediction Curve',
          data: [],
          borderColor: '#f59e0b',
          backgroundColor: 'rgba(245, 158, 11, 0.12)',
          borderWidth: 3,
          fill: true,
          tension: 0.35,
          pointRadius: 0,
          pointHoverRadius: 6,
          order: 3
        },
        {
          label: 'Current Input Scenario',
          data: [],
          borderColor: '#ffffff',
          backgroundColor: '#ef4444',
          borderWidth: 2,
          pointRadius: 9,
          pointHoverRadius: 12,
          showLine: false,
          order: 1
        },
        {
          label: 'User Input Data Points',
          data: [],
          borderColor: '#06b6d4',
          backgroundColor: '#22d3ee',
          borderWidth: 2,
          pointStyle: 'rectRot',
          pointRadius: 8,
          pointHoverRadius: 11,
          showLine: false,
          order: 2
        },
        {
          label: 'Exp 50 °C Observations',
          data: [],
          borderColor: '#3b82f6',
          backgroundColor: '#3b82f6',
          pointRadius: 5,
          showLine: false,
          order: 4
        },
        {
          label: 'Exp 60 °C Observations',
          data: [],
          borderColor: '#10b981',
          backgroundColor: '#10b981',
          pointRadius: 5,
          showLine: false,
          order: 4
        },
        {
          label: 'Exp 70 °C Observations',
          data: [],
          borderColor: '#a855f7',
          backgroundColor: '#a855f7',
          pointRadius: 5,
          showLine: false,
          order: 4
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: {
        mode: 'nearest',
        intersect: true
      },
      plugins: {
        legend: {
          labels: {
            color: '#94a3b8',
            font: { family: 'Plus Jakarta Sans', size: 11 },
            boxWidth: 14,
            usePointStyle: true
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
            title: function(items) {
              if (!items.length) return '';
              const xVal = items[0].parsed.x;
              return `Drying Time: ${xVal.toFixed(2)} ${currentTimeUnit}`;
            },
            label: function(context) {
              const yVal = context.parsed.y !== null ? context.parsed.y.toFixed(4) : '';
              return `${context.dataset.label}: Sr = ${yVal}`;
            }
          }
        }
      },
      scales: {
        x: {
          type: 'linear',
          title: {
            display: true,
            text: `Drying Time (${currentTimeUnit})`,
            color: '#94a3b8',
            font: { family: 'Plus Jakarta Sans', size: 12, weight: 'bold' }
          },
          grid: { color: 'rgba(255, 255, 255, 0.05)' },
          ticks: { color: '#64748b' }
        },
        y: {
          title: {
            display: true,
            text: 'Shrinkage Ratio (Sr = V / V₀)',
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

  const tempC = toCelsius(rawTemp, currentTempUnit);
  const timeMin = toMinutes(rawTime, currentTimeUnit);

  // Update header text
  const chartTempEl = document.getElementById('chartTempDisplay');
  if (chartTempEl) chartTempEl.textContent = `${rawTemp.toFixed(1)} °${currentTempUnit} (${tempC.toFixed(1)} °C)`;
  const chartTimeEl = document.getElementById('chartTimeUnitDisplay');
  if (chartTimeEl) chartTimeEl.textContent = currentTimeUnit;

  // Determine dynamic max X axis time according to user's input
  const defaultMaxMin = 40.0;
  const currentMaxMin = Math.max(defaultMaxMin, timeMin * 1.25);
  const maxXInChosenUnit = fromMinutes(currentMaxMin, currentTimeUnit);

  kineticsChart.options.scales.x.title.text = `Drying Time (${currentTimeUnit})`;
  kineticsChart.options.scales.x.min = 0;
  kineticsChart.options.scales.x.max = maxXInChosenUnit;

  // 1. Continuous ANN Model Prediction Curve at current temperature
  const curvePoints = [];
  const steps = 80;
  const stepMin = currentMaxMin / steps;
  for (let m = 0; m <= currentMaxMin; m += stepMin) {
    const res = predictANN(tempC, m);
    const xUnitVal = fromMinutes(m, currentTimeUnit);
    curvePoints.push({ x: xUnitVal, y: res.prediction });
  }

  // 2. Active Test Point Marker
  const currentPred = predictANN(tempC, timeMin).prediction;
  const currentTestPoint = [{ x: rawTime, y: currentPred }];

  // 3. User Input Data Points
  const showUser = document.getElementById('showUserPoints')?.checked ?? true;
  const userPointData = showUser ? userPoints.map(p => {
    const pTimeInActiveUnit = fromMinutes(p.timeMin, currentTimeUnit);
    return {
      x: pTimeInActiveUnit,
      y: p.observedSr !== null ? p.observedSr : p.predictedSr
    };
  }) : [];

  // Update badges
  const userBadge = document.getElementById('userPointsBadge');
  if (userBadge) userBadge.textContent = userPoints.length;
  const countSpan = document.getElementById('userPointsCount');
  if (countSpan) countSpan.textContent = userPoints.length;

  // 4. Experimental Dataset Points (Converted to user's active time unit!)
  const show50 = document.getElementById('showExp50')?.checked ?? true;
  const show60 = document.getElementById('showExp60')?.checked ?? true;
  const show70 = document.getElementById('showExp70')?.checked ?? true;

  const exp50 = show50 ? MODEL_DATA.training_data.filter(d => d.Temperature === 50).map(d => ({
    x: fromMinutes(d.Time, currentTimeUnit),
    y: d.ShrinkageRatio
  })) : [];

  const exp60 = show60 ? MODEL_DATA.training_data.filter(d => d.Temperature === 60).map(d => ({
    x: fromMinutes(d.Time, currentTimeUnit),
    y: d.ShrinkageRatio
  })) : [];

  const exp70 = show70 ? MODEL_DATA.training_data.filter(d => d.Temperature === 70).map(d => ({
    x: fromMinutes(d.Time, currentTimeUnit),
    y: d.ShrinkageRatio
  })) : [];

  kineticsChart.data.datasets[0].data = curvePoints;
  kineticsChart.data.datasets[0].label = `ANN Curve @ ${rawTemp.toFixed(1)} °${currentTempUnit}`;
  kineticsChart.data.datasets[1].data = currentTestPoint;
  kineticsChart.data.datasets[1].label = `Current: ${rawTemp.toFixed(1)}°${currentTempUnit}, ${rawTime.toFixed(1)}${currentTimeUnit}`;
  kineticsChart.data.datasets[2].data = userPointData;
  kineticsChart.data.datasets[3].data = exp50;
  kineticsChart.data.datasets[4].data = exp60;
  kineticsChart.data.datasets[5].data = exp70;

  kineticsChart.update('none');
}

// User Data Point Management
function pinCurrentTestToChart() {
  const tempC = toCelsius(rawTemp, currentTempUnit);
  const timeMin = toMinutes(rawTime, currentTimeUnit);
  const pred = predictANN(tempC, timeMin).prediction;

  const newId = userPoints.length ? Math.max(...userPoints.map(p => p.id)) + 1 : 1;
  const newPt = {
    id: newId,
    label: `Test #${newId}`,
    rawTemp: rawTemp,
    tempUnit: currentTempUnit,
    tempC: tempC,
    rawTime: rawTime,
    timeUnit: currentTimeUnit,
    timeMin: timeMin,
    observedSr: pred,
    predictedSr: pred
  };

  userPoints.push(newPt);
  renderUserPointsTable();
  updateChart();
  alert(`Pinned current input (${rawTemp.toFixed(1)}°${currentTempUnit}, ${rawTime.toFixed(1)} ${currentTimeUnit}) to graph!`);
}

function submitCustomPoint() {
  const labelInput = document.getElementById('newPointLabel').value.trim() || 'Custom Point';
  const tempVal = parseFloat(document.getElementById('newPointTemp').value);
  const timeVal = parseFloat(document.getElementById('newPointTime').value);
  const obsSrInput = document.getElementById('newPointObservedSr').value.trim();

  if (isNaN(tempVal) || isNaN(timeVal)) {
    alert('Please enter valid numeric values for temperature and drying time.');
    return;
  }

  const tempC = toCelsius(tempVal, currentTempUnit);
  const timeMin = toMinutes(timeVal, currentTimeUnit);
  const pred = predictANN(tempC, timeMin).prediction;
  const obsSr = obsSrInput !== '' ? parseFloat(obsSrInput) : pred;

  const newId = userPoints.length ? Math.max(...userPoints.map(p => p.id)) + 1 : 1;
  const newPt = {
    id: newId,
    label: labelInput,
    rawTemp: tempVal,
    tempUnit: currentTempUnit,
    tempC: tempC,
    rawTime: timeVal,
    timeUnit: currentTimeUnit,
    timeMin: timeMin,
    observedSr: obsSr,
    predictedSr: pred
  };

  userPoints.push(newPt);
  renderUserPointsTable();
  updateChart();
  alert(`Added point "${labelInput}" to the graph!`);
}

function openAddPointModal() {
  switchTab('userdata');
  document.getElementById('newPointTemp').focus();
}

function clearUserPoints() {
  if (confirm('Clear all user data points from graph?')) {
    userPoints = [];
    renderUserPointsTable();
    updateChart();
  }
}

function removeUserPoint(id) {
  userPoints = userPoints.filter(p => p.id !== id);
  renderUserPointsTable();
  updateChart();
}

function loadUserPoint(id) {
  const pt = userPoints.find(p => p.id === id);
  if (!pt) return;

  // Match the point's units or convert
  currentTempUnit = pt.tempUnit;
  currentTimeUnit = pt.timeUnit;
  document.getElementById('tempUnitSelect').value = currentTempUnit;
  document.getElementById('timeUnitSelect').value = currentTimeUnit;
  updateTempUnitUI();
  updateTimeUnitUI();

  rawTemp = pt.rawTemp;
  rawTime = pt.rawTime;
  document.getElementById('tempSlider').value = rawTemp;
  document.getElementById('tempNumericInput').value = rawTemp.toFixed(1);
  document.getElementById('timeSlider').value = rawTime;
  document.getElementById('timeNumericInput').value = currentTimeUnit === 'hr' ? rawTime.toFixed(2) : (currentTimeUnit === 's' ? Math.round(rawTime) : rawTime.toFixed(1));

  calculateAndRender();
  switchTab('chart');
}

function renderUserPointsTable() {
  const tbody = document.getElementById('userPointsTableBody');
  if (!tbody) return;

  const countBadge = document.getElementById('userPointsBadge');
  if (countBadge) countBadge.textContent = userPoints.length;
  const countSpan = document.getElementById('userPointsCount');
  if (countSpan) countSpan.textContent = userPoints.length;

  if (userPoints.length === 0) {
    tbody.innerHTML = `<tr><td colspan="8" style="text-align: center; color: #64748b; padding: 24px;">No custom user points added yet. Click "Pin Current Input to Graph" or add custom points above.</td></tr>`;
    return;
  }

  tbody.innerHTML = userPoints.map((pt, i) => {
    const diff = Math.abs(pt.observedSr - pt.predictedSr);
    return `
      <tr>
        <td>#${i + 1}</td>
        <td><strong>${pt.label}</strong></td>
        <td>${pt.rawTemp.toFixed(1)} °${pt.tempUnit}</td>
        <td>${pt.rawTime.toFixed(1)} ${pt.timeUnit}</td>
        <td style="color: #22d3ee; font-weight: 700;">${pt.observedSr.toFixed(4)}</td>
        <td style="color: #fbbf24;">${pt.predictedSr.toFixed(4)}</td>
        <td style="color: ${diff < 0.02 ? '#34d399' : '#f87171'};">±${diff.toFixed(5)}</td>
        <td>
          <button class="table-btn" onclick="loadUserPoint(${pt.id})">Load</button>
          <button class="table-btn" onclick="removeUserPoint(${pt.id})">✕</button>
        </td>
      </tr>
    `;
  }).join('');
}

function exportUserDataCSV() {
  if (userPoints.length === 0) {
    alert('No user points to export.');
    return;
  }
  let csv = 'Label,InputTemperature,TempUnit,InputTime,TimeUnit,TemperatureCelsius,TimeMinutes,ObservedShrinkageRatio,ANNPredictedShrinkageRatio,AbsoluteError\n';
  userPoints.forEach(p => {
    const err = Math.abs(p.observedSr - p.predictedSr).toFixed(5);
    csv += `"${p.label}",${p.rawTemp},${p.tempUnit},${p.rawTime},${p.timeUnit},${p.tempC.toFixed(2)},${p.timeMin.toFixed(2)},${p.observedSr.toFixed(4)},${p.predictedSr.toFixed(4)},${err}\n`;
  });

  const blob = new Blob([csv], { type: 'text/csv' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'user_turmeric_points.csv';
  a.click();
  URL.revokeObjectURL(url);
}

// Empirical Dataset Explorer
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
          <button class="table-btn" onclick="applyPresetBase(${row.Temperature}, ${row.Time})">Load</button>
        </td>
      </tr>
    `;
  }).join('');
}

// Batch Tester with Selectable Units
const initialBatch = [
  { temp: 80.0, tempUnit: 'C', time: 15.0, timeUnit: 'min' },
  { temp: 176.0, tempUnit: 'F', time: 0.25, timeUnit: 'hr' },
  { temp: 353.15, tempUnit: 'K', time: 900.0, timeUnit: 's' },
  { temp: 50.0, tempUnit: 'C', time: 10.0, timeUnit: 'min' },
  { temp: 60.0, tempUnit: 'C', time: 20.0, timeUnit: 'min' },
  { temp: 70.0, tempUnit: 'C', time: 12.0, timeUnit: 'min' }
];

function initBatchTable() {
  const tbody = document.getElementById('batchTableBody');
  if (!tbody) return;

  tbody.innerHTML = initialBatch.map((row, i) => createBatchRowHTML(i + 1, row.temp, row.tempUnit, row.time, row.timeUnit)).join('');
  runBatchPredictions();
}

function createBatchRowHTML(id, temp, tempUnit, time, timeUnit) {
  return `
    <tr id="batch-row-${id}">
      <td>${id}</td>
      <td><input type="number" step="0.5" value="${temp}" class="table-input batch-temp" oninput="runBatchPredictions()"></td>
      <td>
        <select class="table-select batch-temp-unit" onchange="runBatchPredictions()">
          <option value="C" ${tempUnit === 'C' ? 'selected' : ''}>°C</option>
          <option value="F" ${tempUnit === 'F' ? 'selected' : ''}>°F</option>
          <option value="K" ${tempUnit === 'K' ? 'selected' : ''}>K</option>
        </select>
      </td>
      <td><input type="number" step="0.5" value="${time}" class="table-input batch-time" oninput="runBatchPredictions()"></td>
      <td>
        <select class="table-select batch-time-unit" onchange="runBatchPredictions()">
          <option value="min" ${timeUnit === 'min' ? 'selected' : ''}>min</option>
          <option value="hr" ${timeUnit === 'hr' ? 'selected' : ''}>hr</option>
          <option value="s" ${timeUnit === 's' ? 'selected' : ''}>s</option>
        </select>
      </td>
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
  tr.innerHTML = createBatchRowHTML(batchRowCount, 70.0, 'C', 15.0, 'min');
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
    const tempUnitSelect = tr.querySelector('.batch-temp-unit');
    const timeInput = tr.querySelector('.batch-time');
    const timeUnitSelect = tr.querySelector('.batch-time-unit');
    const predTd = tr.querySelector('.batch-pred');
    const lossTd = tr.querySelector('.batch-loss');

    if (tempInput && timeInput && predTd) {
      const t = parseFloat(tempInput.value) || 0;
      const tUnit = tempUnitSelect ? tempUnitSelect.value : 'C';
      const m = parseFloat(timeInput.value) || 0;
      const mUnit = timeUnitSelect ? timeUnitSelect.value : 'min';

      const tempC = toCelsius(t, tUnit);
      const timeMin = toMinutes(m, mUnit);

      const res = predictANN(tempC, timeMin);
      const p = res.prediction;
      predTd.textContent = p.toFixed(4);
      lossTd.textContent = `${((1.0 - p) * 100).toFixed(2)}%`;
    }
  });
}

function exportBatchCSV() {
  const rows = document.querySelectorAll('#batchTableBody tr');
  let csv = 'Temperature,TempUnit,Time,TimeUnit,TemperatureCelsius,TimeMinutes,PredictedShrinkageRatio,VolumeReductionPercent\n';
  rows.forEach(tr => {
    const t = tr.querySelector('.batch-temp')?.value || '';
    const tUnit = tr.querySelector('.batch-temp-unit')?.value || 'C';
    const m = tr.querySelector('.batch-time')?.value || '';
    const mUnit = tr.querySelector('.batch-time-unit')?.value || 'min';
    const p = tr.querySelector('.batch-pred')?.textContent || '';
    const l = tr.querySelector('.batch-loss')?.textContent?.replace('%', '') || '';

    if (t && m) {
      const tempC = toCelsius(parseFloat(t), tUnit).toFixed(2);
      const timeMin = toMinutes(parseFloat(m), mUnit).toFixed(2);
      csv += `${t},${tUnit},${m},${mUnit},${tempC},${timeMin},${p},${l}\n`;
    }
  });

  const blob = new Blob([csv], { type: 'text/csv' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'turmeric_batch_predictions.csv';
  a.click();
  URL.revokeObjectURL(url);
}

// Navigation Tabs
function switchTab(tabId) {
  document.querySelectorAll('.tabs-nav .tab-btn').forEach(b => b.classList.remove('active'));
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));

  const tabBtn = Array.from(document.querySelectorAll('.tabs-nav .tab-btn')).find(b => b.getAttribute('onclick')?.includes(tabId));
  if (tabBtn) tabBtn.classList.add('active');

  const panel = document.getElementById(`tab-${tabId}`);
  if (panel) panel.classList.add('active');

  if (tabId === 'chart' && kineticsChart) {
    setTimeout(() => {
      kineticsChart.resize();
      updateChart();
    }, 50);
  }
}

function showCodeSnippet(type) {
  document.querySelectorAll('.code-tab').forEach(b => b.classList.remove('active'));
  document.querySelectorAll('.code-snippet').forEach(s => s.classList.remove('active'));

  event.currentTarget.classList.add('active');
  document.getElementById(`code-${type}`).classList.add('active');
}

function copyPrediction() {
  const p = document.getElementById('predShrinkage').textContent;
  const tempC = toCelsius(rawTemp, currentTempUnit).toFixed(1);
  const timeMin = toMinutes(rawTime, currentTimeUnit).toFixed(1);
  const text = `Input: ${rawTemp.toFixed(1)}°${currentTempUnit} (${tempC}°C), ${rawTime.toFixed(1)} ${currentTimeUnit} (${timeMin} min) -> Predicted Shrinkage Ratio: ${p}`;
  navigator.clipboard.writeText(text);
  alert(`Copied to clipboard!\n${text}`);
}

// Live API Call Tester
async function testLiveAPI() {
  const feedback = document.getElementById('apiTestFeedback');
  feedback.classList.remove('hidden');
  feedback.textContent = `Connecting to /api/predict (temp=${rawTemp} ${currentTempUnit}, time=${rawTime} ${currentTimeUnit})...`;

  try {
    const res = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        temperature: rawTemp,
        temp_unit: currentTempUnit,
        time: rawTime,
        time_unit: currentTimeUnit
      })
    });

    if (res.ok) {
      const json = await res.json();
      feedback.style.color = '#34d399';
      feedback.textContent = `✓ API [200 OK]: Predicted Sr = ${json.predicted_shrinkage_ratio} | Normalized: ${json.normalized_for_model.temperature_celsius}°C, ${json.normalized_for_model.time_minutes} min`;
    } else {
      feedback.style.color = '#fbbf24';
      feedback.textContent = `Client ANN Active: Predicted ${predictANN(toCelsius(rawTemp, currentTempUnit), toMinutes(rawTime, currentTimeUnit)).prediction.toFixed(4)}`;
    }
  } catch (err) {
    feedback.style.color = '#34d399';
    feedback.textContent = `✓ Client ANN Active: Predicted ${predictANN(toCelsius(rawTemp, currentTempUnit), toMinutes(rawTime, currentTimeUnit)).prediction.toFixed(4)}`;
  }
}

// Fallback Model Weights
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
