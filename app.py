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

def convert_temp_to_celsius(val: float, unit: str = 'C') -> float:
    unit = str(unit).strip().upper()
    if unit in ['F', 'FAHRENHEIT', '°F']:
        return (val - 32.0) * 5.0 / 9.0
    elif unit in ['K', 'KELVIN']:
        return val - 273.15
    return val

def convert_time_to_minutes(val: float, unit: str = 'min') -> float:
    unit = str(unit).strip().lower()
    if unit in ['h', 'hr', 'hrs', 'hour', 'hours']:
        return val * 60.0
    elif unit in ['s', 'sec', 'secs', 'second', 'seconds']:
        return val / 60.0
    return val

def run_forward_pass(temp_c: float, time_min: float):
    # If scikit-learn model is available, can use it
    if sklearn_model is not None and sklearn_scaler is not None:
        scaled = sklearn_scaler.transform([[temp_c, time_min]])
        return float(sklearn_model.predict(scaled)[0])
    
    # Standalone exact forward pass
    mean = model_data['scaler']['mean']
    scale = model_data['scaler']['scale']
    x0 = (temp_c - mean[0]) / scale[0]
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
            temp_unit = request.args.get('temp_unit', 'C')
            time_unit = request.args.get('time_unit', 'min')

            if temp_str is not None and time_str is not None:
                raw_temp = float(temp_str)
                raw_time = float(time_str)
                temp_c = convert_temp_to_celsius(raw_temp, temp_unit)
                time_min = convert_time_to_minutes(raw_time, time_unit)

                prediction = run_forward_pass(temp_c, time_min)
                return jsonify({
                    "input": {
                        "temperature": raw_temp,
                        "temp_unit": temp_unit,
                        "time": raw_time,
                        "time_unit": time_unit
                    },
                    "normalized_for_model": {
                        "temperature_celsius": round(temp_c, 2),
                        "time_minutes": round(time_min, 2)
                    },
                    "predicted_shrinkage_ratio": round(prediction, 4),
                    "raw_prediction": prediction,
                    "volume_reduction_percent": round((1.0 - prediction) * 100, 2),
                    "model": "MLPRegressor (8, 4)",
                    "r2_score": model_data['metrics']['r2_score']
                })
            return jsonify({
                "status": "online",
                "model": model_data.get("model_type"),
                "architecture": model_data.get("architecture"),
                "metrics": model_data.get("metrics"),
                "supported_units": {
                    "temperature": ["C", "F", "K"],
                    "time": ["min", "hr", "s"]
                }
            })

        data = request.get_json(force=True)
        if isinstance(data, list):
            results = []
            for item in data:
                raw_temp = float(item.get('temperature', item.get('temp', 0)))
                raw_time = float(item.get('time', item.get('time_min', 0)))
                temp_unit = item.get('temp_unit', 'C')
                time_unit = item.get('time_unit', 'min')
                
                temp_c = convert_temp_to_celsius(raw_temp, temp_unit)
                time_min = convert_time_to_minutes(raw_time, time_unit)
                pred = run_forward_pass(temp_c, time_min)
                results.append({
                    "temperature": raw_temp,
                    "temp_unit": temp_unit,
                    "time": raw_time,
                    "time_unit": time_unit,
                    "temperature_celsius": round(temp_c, 2),
                    "time_minutes": round(time_min, 2),
                    "predicted_shrinkage_ratio": round(pred, 4),
                    "raw_prediction": pred,
                    "volume_reduction_percent": round((1.0 - pred) * 100, 2)
                })
            return jsonify({"count": len(results), "predictions": results})
        else:
            raw_temp = float(data.get('temperature', data.get('temp', 0)))
            raw_time = float(data.get('time', data.get('time_min', 0)))
            temp_unit = data.get('temp_unit', 'C')
            time_unit = data.get('time_unit', 'min')
            
            temp_c = convert_temp_to_celsius(raw_temp, temp_unit)
            time_min = convert_time_to_minutes(raw_time, time_unit)
            
            prediction = run_forward_pass(temp_c, time_min)
            return jsonify({
                "input": {
                    "temperature": raw_temp,
                    "temp_unit": temp_unit,
                    "time": raw_time,
                    "time_unit": time_unit
                },
                "normalized_for_model": {
                    "temperature_celsius": round(temp_c, 2),
                    "time_minutes": round(time_min, 2)
                },
                "predicted_shrinkage_ratio": round(prediction, 4),
                "raw_prediction": prediction,
                "volume_reduction_percent": round((1.0 - prediction) * 100, 2),
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
