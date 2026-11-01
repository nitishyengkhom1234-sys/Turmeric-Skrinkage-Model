# 🌿 Turmeric Drying Shrinkage Ratio Predictor (ANN MLP Model)

An interactive Artificial Neural Network (ANN) web application for testing and predicting the **Shrinkage Ratio ($S_r = V / V_0$)** of turmeric rhizomes during convective hot-air drying. Built with Python, Scikit-Learn, and modern vanilla JavaScript with dynamic kinetic curve visualization, zero-dependency serverless inference, and direct Vercel deployment.

[![Accuracy](https://img.shields.io/badge/Model%20Accuracy%20R²-0.99661-success?style=for-the-badge)](https://github.com)
[![Architecture](https://img.shields.io/badge/Architecture-ANN%20(8%2C%204)%20tanh-amber?style=for-the-badge)](https://github.com)
[![Deploy](https://img.shields.io/badge/Deployment-Vercel%20Serverless-black?style=for-the-badge&logo=vercel)](https://vercel.com)

---

## 📌 Overview

During convective thermal drying of turmeric rhizomes, moisture removal triggers tissue contraction and volumetric shrinkage. Predicting the shrinkage ratio as a function of **Temperature (°C)** and **Drying Duration (min)** is critical for food processing optimization, energy management, and quality control.

This web application wraps an optimized **Multi-Layer Perceptron Regressor (`MLPRegressor`)** with:
- **Instant Client-Side & Server-Side Prediction**: Instant response with zero cold-boot delays.
- **Dynamic Drying Kinetics Chart**: Real-time curve of $S_r$ vs Time at any selected temperature with experimental scatter plots for 50°C, 60°C, and 70°C.
- **Morphological Rhizome Visualizer**: Visual simulation of physical shrinkage.
- **Batch Evaluation & CSV Export**: Bulk testing for laboratory experiments.
- **Production REST API**: Serverless `/api/predict` endpoint ready for integration.

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
Input: Temperature = 80.0 °C, Time = 15.0 min
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

## ⚡ Production REST API Reference

### `POST /api/predict`
Calculates the predicted shrinkage ratio for a given temperature and drying duration.

#### Request (JSON)
```json
{
  "temperature": 80.0,
  "time": 15.0
}
```

#### Response (200 OK)
```json
{
  "temperature": 80.0,
  "time": 15.0,
  "predicted_shrinkage_ratio": 0.3593,
  "raw_prediction": 0.3592706320668791,
  "model": "MLPRegressor (8, 4)",
  "r2_score": 0.99661
}
```

#### cURL Example
```bash
curl -X POST https://<your-vercel-domain>.vercel.app/api/predict \
  -H "Content-Type: application/json" \
  -d '{"temperature": 80.0, "time": 15.0}'
```

#### Python Example
```python
import requests

url = "https://<your-vercel-domain>.vercel.app/api/predict"
payload = {"temperature": 80.0, "time": 15.0}
response = requests.post(url, json=payload)
print(response.json())
```

---

## 🌐 Deploying to Vercel

### Option 1: Automatic Deployment via GitHub (Recommended)
1. Push this repository to GitHub:
   ```bash
   git init
   git add .
   git commit -m "Initial commit of Turmeric ANN predictor"
   git remote add origin https://github.com/nitishyengkhom1234-sys/Turmeric-Skrinkage-Model.git
   git branch -M main
   git push -u origin main
   ```
2. Go to [vercel.com](https://vercel.com) and log in.
3. Click **Add New...** > **Project**.
4. Import your `turmeric-shrinkage-ann` repository.
5. Click **Deploy**. Vercel will build and serve both the static web frontend and the `/api/predict` serverless functions automatically!

### Option 2: Deploy directly via Vercel CLI
```bash
vercel login
vercel --prod
```

---

## 📂 Project Structure

```
Turmeric/
├── api/
│   └── predict.py          # Vercel serverless Python API handler
├── index.html              # Modern, responsive testing UI
├── style.css               # Clean styling with turmeric amber palette
├── app.js                  # Dynamic UI, Chart.js curves, forward pass logic
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
