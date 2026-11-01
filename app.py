import os
import json
import webbrowser
from threading import Timer
from flask import Flask, request, jsonify, send_from_directory

app = Flask(__name__, static_folder='.')

# Load model weights JSON
with open('model_weights.json', 'r') as f:
    model_data = json.load(f)

# Also try loading scikit-learn model if available
sklearn_model = None
sklearn_scaler = None
try:
    import joblib
    if os.path.exists('turmeric_model.joblib') and os.path.exists('scaler.joblib'):
        sklearn_model = joblib.load('turmeric_model.joblib')
        sklearn_scaler = joblib.load('scaler.joblib')
        print("Loaded scikit-learn model and scaler from joblib.")
except Exception as e:
    print(f"Running in pure math inference mode: {e}")

def run_forward_pass(temp: float, time_min: float):
    # If scikit-learn model is available, can use it
    if sklearn_model is not None and sklearn_scaler is not None:
        scaled = sklearn_scaler.transform([[temp, time_min]])
        return float(sklearn_model.predict(scaled)[0])
    
    # Standalone exact forward pass
    mean = model_data['scaler']['mean']
    scale = model_data['scaler']['scale']
    x0 = (temp - mean[0]) / scale[0]
    x1 = (time_min - mean[1]) / scale[1]
    
    import math
    c0 = model_data['weights']['coefs'][0]
    b0 = model_data['weights']['intercepts'][0]
    h1 = [math.tanh(x0 * c0[0][j] + x1 * c0[1][j] + b0[j]) for j in range(8)]
    
    c1 = model_data['weights']['coefs'][1]
    b1 = model_data['weights']['intercepts'][1]
    h2 = [math.tanh(sum(h1[k] * c1[k][j] for k in range(8)) + b1[j]) for j in range(4)]
    
    c2 = model_data['weights']['coefs'][2]
    b2 = model_data['weights']['intercepts'][2]
    return float(sum(h2[k] * c2[k][0] for k in range(4)) + b2[0])

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/<path:path>')
def static_files(path):
    return send_from_directory('.', path)

@app.route('/api/predict', methods=['GET', 'POST'])
def predict_endpoint():
    try:
        if request.method == 'GET':
            temp_str = request.args.get('temp') or request.args.get('temperature')
            time_str = request.args.get('time')
            if temp_str is not None and time_str is not None:
                temp = float(temp_str)
                time_min = float(time_str)
                prediction = run_forward_pass(temp, time_min)
                return jsonify({
                    "temperature": temp,
                    "time": time_min,
                    "predicted_shrinkage_ratio": round(prediction, 4),
                    "raw_prediction": prediction,
                    "model": "MLPRegressor (8, 4)",
                    "r2_score": model_data['metrics']['r2_score']
                })
            return jsonify({
                "status": "online",
                "model": model_data.get("model_type"),
                "architecture": model_data.get("architecture"),
                "metrics": model_data.get("metrics")
            })

        data = request.get_json(force=True)
        if isinstance(data, list):
            results = []
            for item in data:
                temp = float(item.get('temperature', item.get('temp', 0)))
                time_min = float(item.get('time', item.get('time_min', 0)))
                pred = run_forward_pass(temp, time_min)
                results.append({
                    "temperature": temp,
                    "time": time_min,
                    "predicted_shrinkage_ratio": round(pred, 4),
                    "raw_prediction": pred
                })
            return jsonify({"count": len(results), "predictions": results})
        else:
            temp = float(data.get('temperature', data.get('temp', 0)))
            time_min = float(data.get('time', data.get('time_min', 0)))
            prediction = run_forward_pass(temp, time_min)
            return jsonify({
                "temperature": temp,
                "time": time_min,
                "predicted_shrinkage_ratio": round(prediction, 4),
                "raw_prediction": prediction,
                "model": "MLPRegressor (8, 4)",
                "r2_score": model_data['metrics']['r2_score']
            })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

def open_browser():
    webbrowser.open_new('http://127.0.0.1:5000/')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"\n=======================================================")
    print(f" Turmeric Drying Shrinkage Ratio ANN Predictor")
    print(f" Server running at: http://127.0.0.1:{port}/")
    print(f"=======================================================\n")
    Timer(1.5, open_browser).start()
    app.run(host='0.0.0.0', port=port, debug=False)
