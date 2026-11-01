import json
import math
import os
from http.server import BaseHTTPRequestHandler

# Load model weights
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS_PATH = os.path.join(BASE_DIR, 'model_weights.json')

_model_cache = None

def get_model():
    global _model_cache
    if _model_cache is None:
        if os.path.exists(WEIGHTS_PATH):
            with open(WEIGHTS_PATH, 'r') as f:
                _model_cache = json.load(f)
        else:
            with open('model_weights.json', 'r') as f:
                _model_cache = json.load(f)
    return _model_cache

def convert_temp_to_celsius(val: float, unit: str = 'C') -> float:
    unit = str(unit).strip().upper()
    if unit in ['F', 'FAHRENHEIT', '°F']:
        return (val - 32.0) * 5.0 / 9.0
    elif unit in ['K', 'KELVIN']:
        return val - 273.15
    return val  # Default Celsius

def convert_time_to_minutes(val: float, unit: str = 'min') -> float:
    unit = str(unit).strip().lower()
    if unit in ['h', 'hr', 'hrs', 'hour', 'hours']:
        return val * 60.0
    elif unit in ['s', 'sec', 'secs', 'second', 'seconds']:
        return val / 60.0
    return val  # Default minutes

def predict_single(temp: float, time_min: float):
    model = get_model()
    
    # 1. Scale input features (StandardScaler)
    mean = model['scaler']['mean']
    scale = model['scaler']['scale']
    x0 = (float(temp) - mean[0]) / scale[0]
    x1 = (float(time_min) - mean[1]) / scale[1]
    
    # 2. Hidden Layer 1: 2 -> 8 (tanh activation)
    c0 = model['weights']['coefs'][0]
    b0 = model['weights']['intercepts'][0]
    h1 = [math.tanh(x0 * c0[0][j] + x1 * c0[1][j] + b0[j]) for j in range(8)]
    
    # 3. Hidden Layer 2: 8 -> 4 (tanh activation)
    c1 = model['weights']['coefs'][1]
    b1 = model['weights']['intercepts'][1]
    h2 = [math.tanh(sum(h1[k] * c1[k][j] for k in range(8)) + b1[j]) for j in range(4)]
    
    # 4. Output Layer: 4 -> 1 (identity activation)
    c2 = model['weights']['coefs'][2]
    b2 = model['weights']['intercepts'][2]
    pred = sum(h2[k] * c2[k][0] for k in range(4)) + b2[0]
    
    return float(pred)

class handler(BaseHTTPRequestHandler):
    def _send_cors_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')

    def do_OPTIONS(self):
        self.send_response(204)
        self._send_cors_headers()
        self.end_headers()

    def do_GET(self):
        try:
            from urllib.parse import urlparse, parse_qs
            query = parse_qs(urlparse(self.path).query)
            
            if ('temp' in query or 'temperature' in query) and 'time' in query:
                raw_temp = float(query.get('temp', query.get('temperature'))[0])
                raw_time = float(query['time'][0])
                temp_unit = query.get('temp_unit', ['C'])[0]
                time_unit = query.get('time_unit', ['min'])[0]
                
                temp_c = convert_temp_to_celsius(raw_temp, temp_unit)
                time_min = convert_time_to_minutes(raw_time, time_unit)
                
                prediction = predict_single(temp_c, time_min)
                response = {
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
                    "r2_score": 0.99661
                }
            else:
                model = get_model()
                response = {
                    "status": "online",
                    "model": model.get("model_type", "MLPRegressor"),
                    "architecture": model.get("architecture"),
                    "metrics": model.get("metrics"),
                    "usage": {
                        "post": "Send JSON {'temperature': 80.0, 'temp_unit': 'C', 'time': 15.0, 'time_unit': 'min'}",
                        "supported_units": {
                            "temperature": ["C", "F", "K"],
                            "time": ["min", "hr", "s"]
                        }
                    }
                }
            
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self._send_cors_headers()
            self.end_headers()
            self.wfile.write(json.dumps(response, indent=2).encode('utf-8'))
        except Exception as e:
            self.send_response(500)
            self.send_header('Content-Type', 'application/json')
            self._send_cors_headers()
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode('utf-8'))

    def do_POST(self):
        try:
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length).decode('utf-8')
            payload = json.loads(body)
            
            if isinstance(payload, list):
                # Batch predictions
                results = []
                for item in payload:
                    raw_temp = float(item.get('temperature', item.get('temp', 0)))
                    raw_time = float(item.get('time', item.get('time_min', 0)))
                    temp_unit = item.get('temp_unit', 'C')
                    time_unit = item.get('time_unit', 'min')
                    
                    temp_c = convert_temp_to_celsius(raw_temp, temp_unit)
                    time_min = convert_time_to_minutes(raw_time, time_unit)
                    
                    pred = predict_single(temp_c, time_min)
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
                response = {"count": len(results), "predictions": results}
            else:
                raw_temp = float(payload.get('temperature', payload.get('temp', 0)))
                raw_time = float(payload.get('time', payload.get('time_min', 0)))
                temp_unit = payload.get('temp_unit', 'C')
                time_unit = payload.get('time_unit', 'min')
                
                temp_c = convert_temp_to_celsius(raw_temp, temp_unit)
                time_min = convert_time_to_minutes(raw_time, time_unit)
                
                prediction = predict_single(temp_c, time_min)
                response = {
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
                    "r2_score": 0.99661
                }
                
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self._send_cors_headers()
            self.end_headers()
            self.wfile.write(json.dumps(response, indent=2).encode('utf-8'))
        except Exception as e:
            self.send_response(400)
            self.send_header('Content-Type', 'application/json')
            self._send_cors_headers()
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode('utf-8'))
