<p align="center">
  <img src="logo.png" width="180" alt="Turmeric Drying Modulus & Kinetics Study Logo">
</p>

# 🌿 Turmeric Drying Shrinkage Ratio Predictor (ANN MLP Model)

An interactive Artificial Neural Network (ANN) web application for testing and predicting the **Shrinkage Ratio ($S_r = V / V_0$)** of turmeric rhizomes during convective hot-air drying. Built with Python, Scikit-Learn, and modern vanilla JavaScript with dynamic kinetic curve visualization, multi-unit input conversion, zero-dependency serverless inference, and direct Vercel deployment.

[![Accuracy](https://img.shields.io/badge/Model%20Accuracy%20R²-0.99661-success?style=for-the-badge)](https://github.com/nitishyengkhom1234-sys/Turmeric-Skrinkage-Model)
[![Architecture](https://img.shields.io/badge/Architecture-ANN%20(8%2C%204)%20tanh-amber?style=for-the-badge)](https://github.com/nitishyengkhom1234-sys/Turmeric-Skrinkage-Model)
[![Deploy](https://img.shields.io/badge/Deployment-Vercel%20Serverless-black?style=for-the-badge&logo=vercel)](https://vercel.com)

---

## 📌 Features

- **Optional Units for Temperature & Time**:
  - **Temperature Units**: Celsius (`°C`), Fahrenheit (`°F`), and Kelvin (`K`).
  - **Time Units**: Minutes (`min`), Hours (`hr`), and Seconds (`s`).
  - Automatic live conversion and validation against the model's standardized input space.
- **Dynamic Kinetics Graph (User-Data Driven)**:
  - Real-time drying curve of $S_r$ vs Time generated at the user's selected temperature.
  - The graph's X-axis dynamically scales and reflects the user's active time unit (`min`, `hr`, or `s`).
  - **User Data Points Overlay**: Pin current test scenarios or input laboratory observations directly onto the chart to compare empirical measurements with the ANN model.
  - Toggleable experimental data overlays for 50°C, 60°C, and 70°C drying trials.
- **Morphological Rhizome Visualizer**: Visual simulation showing cross-sectional contraction ($S_r^{1/2}$).
- **Batch Evaluation & CSV Export**: Bulk prediction table with per-row unit selectors.
- **Zero-Dependency Serverless REST API**: Production `/api/predict` endpoint deployable on Vercel with zero heavy wheel overhead.

---

## 🧠 Model Specifications & Performance

- **Algorithm**: Artificial Neural Network / Multi-Layer Perceptron (`MLPRegressor`)
- **Input Features (2)**: 
  - `Temperature` (°C)
  - `Time` (minutes)
- **Target Output (1)**: 
  - `ShrinkageRatio` ($S_r = V / V_0$)
- **Network Topology**:
  - Input Layer: 2 neurons
  - Hidden Layer 1: 8 neurons (Activation: `tanh`)
  - Hidden Layer 2: 4 neurons (Activation: `tanh`)
  - Output Layer: 1 neuron (Activation: `identity`)
- **Optimization Solver**: L-BFGS (Quasi-Newton method)
- **Data Scaling**: `StandardScaler` ($\mu, \sigma$ normalization)
- **R² Score**: **0.99661** (99.66% variance explained)
- **Mean Squared Error (MSE)**: **0.000114**
- **Root Mean Squared Error (RMSE)**: **0.010654**

### Benchmark Test Case
```
Input: Temperature = 80.0 °C (176.0 °F / 353.15 K), Time = 15.0 min (0.25 hr / 900 s)
--> Predicted Shrinkage Ratio: 0.3593 (64.07% Volume Reduction)
```

---

## 🚀 Running Locally

### 1. Clone the repository
```bash
git clone https://github.com/nitishyengkhom1234-sys/Turmeric-Skrinkage-Model.git
cd Turmeric-Skrinkage-Model
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the application
```bash
python app.py
```
This automatically starts the server at `http://127.0.0.1:5000/` and opens your browser.

---

## ⚡ Production REST API Reference (Unit-Aware)

### `POST /api/predict`
Calculates the predicted shrinkage ratio. Temperature and time units are optional and default to `°C` and `min`.

#### Request with Optional Units (JSON)
```json
{
  "temperature": 176.0,
  "temp_unit": "F",
  "time": 0.25,
  "time_unit": "hr"
}
```

#### Response (200 OK)
```json
{
  "input": {
    "temperature": 176.0,
    "temp_unit": "F",
    "time": 0.25,
    "time_unit": "hr"
  },
  "normalized_for_model": {
    "temperature_celsius": 80.0,
    "time_minutes": 15.0
  },
  "predicted_shrinkage_ratio": 0.3593,
  "raw_prediction": 0.3592706320668791,
  "volume_reduction_percent": 64.07,
  "model": "MLPRegressor (8, 4)",
  "r2_score": 0.99661
}
```

#### cURL Example
```bash
curl -X POST https://turmeric-skrinkage-model.vercel.app/api/predict \
  -H "Content-Type: application/json" \
  -d '{"temperature": 176.0, "temp_unit": "F", "time": 0.25, "time_unit": "hr"}'
```

---

## 🌐 Deploying to Vercel

### Automatic Deployment via GitHub
1. Go to [vercel.com/new](https://vercel.com/new) and log in with your GitHub account.
2. Under **"Import Git Repository"**, select **`Turmeric-Skrinkage-Model`**.
3. Click **Deploy**. Vercel will immediately build and serve both the static web frontend and the `/api/predict` serverless functions.

---

## 📂 Project Structure

```
Turmeric/
├── api/
│   └── predict.py          # Unit-aware Vercel serverless Python API handler
├── index.html              # Modern, responsive UI with unit selection & user-data graph
├── style.css               # Clean styling with turmeric amber palette
├── app.js                  # Dynamic UI, unit conversions, Chart.js curves & user data
├── model_weights.json      # Scaler parameters, trained neural weights & metadata
├── turmeric_model.joblib   # Trained Scikit-Learn MLPRegressor binary
├── scaler.joblib           # Fitted StandardScaler binary
├── turmeric_data.csv       # Experimental drying dataset
├── train_model.py          # Model training, evaluation & export script
├── app.py                  # Local development Flask web server
├── requirements.txt        # Local Python package requirements
├── vercel.json             # Vercel deployment configuration
├── .vercelignore           # Prevents uploading heavy PDF research docs
└── .gitignore              # Standard git ignore rules
```

---

## 📜 License
MIT License. Built for experimental drying kinetics research and predictive modeling.
