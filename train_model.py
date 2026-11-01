import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import joblib

def train_and_export():
    # 1. Load data
    df = pd.read_csv('turmeric_data.csv')

    # 2. Define X (Inputs) and y (Target)
    X = df[['Temperature', 'Time']]
    y = df['ShrinkageRatio']

    # 3. Split data (identical to user code)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. Scale the data
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)

    # 5. Build and train model with scaled data
    ann_model = MLPRegressor(
        hidden_layer_sizes=(8, 4), 
        activation='tanh', 
        solver='lbfgs', 
        max_iter=5000, 
        random_state=42
    )
    ann_model.fit(X_train_scaled, y_train)

    # 6. Evaluation metrics
    y_pred_test = ann_model.predict(X_test_scaled)
    r2 = float(r2_score(y_test, y_pred_test))
    mse = float(mean_squared_error(y_test, y_pred_test))
    rmse = float(np.sqrt(mse))

    print(f"R-squared Score (R2): {r2:.5f}")
    print(f"Mean Squared Error (MSE): {mse:.6f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.6f}")

    # 7. Trial & Testing Section
    my_test_input = [[80.0, 15.0]]
    scaled_input = scaler_X.transform(my_test_input)
    prediction = ann_model.predict(scaled_input)

    print("--- Custom Prediction Result ---")
    print(f"Testing Temperature: {my_test_input[0][0]} °C")
    print(f"Testing Time: {my_test_input[0][1]} minutes")
    print(f"--> Predicted Shrinkage Ratio: {prediction[0]:.4f}")

    # Save joblib binaries
    joblib.dump(ann_model, 'turmeric_model.joblib')
    joblib.dump(scaler_X, 'scaler.joblib')
    print("Saved turmeric_model.joblib and scaler.joblib")

    # Export weights and architecture to JSON for ultra-fast zero-dependency evaluation
    weights_data = {
        "model_type": "MLPRegressor",
        "architecture": {
            "input_dim": 2,
            "hidden_layers": [8, 4],
            "output_dim": 1,
            "activation": "tanh",
            "solver": "lbfgs",
            "random_state": 42
        },
        "metrics": {
            "r2_score": r2,
            "mse": mse,
            "rmse": rmse,
            "sample_count": len(df),
            "train_count": len(X_train),
            "test_count": len(X_test)
        },
        "scaler": {
            "mean": scaler_X.mean_.tolist(),
            "scale": scaler_X.scale_.tolist(),
            "var": scaler_X.var_.tolist()
        },
        "weights": {
            "coefs": [c.tolist() for c in ann_model.coefs_],
            "intercepts": [i.tolist() for i in ann_model.intercepts_]
        },
        "training_data": df.to_dict(orient='records')
    }

    with open('model_weights.json', 'w') as f:
        json.dump(weights_data, f, indent=2)
    print("Saved model_weights.json")

if __name__ == '__main__':
    train_and_export()
