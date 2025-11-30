# app.py - Flask API to serve your ML model predictions

from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import os

app = Flask(__name__)

# Global model and metrics
model = None
metrics_text = ""

def load_model():
    global model, metrics_text
    # Absolute path relative to app.py
    model_path = os.path.join(os.path.dirname(__file__), 'saved_model')
    print("Looking for model at:", model_path)

    if os.path.exists(model_path):
        try:
            model = tf.keras.models.load_model(model_path)
            print("Model loaded successfully!")

            # Optional: calculate metrics on some test data
            # Update X_test and y_test according to your dataset
            X_test = np.arange(40, 100, 4).reshape(-1, 1)
            y_test = np.arange(110, 170, 4).reshape(-1, 1)

            y_preds = model.predict(X_test, verbose=0)
            mae_1 = float(tf.reduce_mean(tf.keras.losses.mean_absolute_error(y_test, y_preds)).numpy())
            mse_1 = float(tf.reduce_mean(tf.keras.losses.mean_squared_error(y_test, y_preds)).numpy())
            metrics_text = f"Mean Absolute Error = {mae_1:.2f}, Mean Squared Error = {mse_1:.2f}"

        except Exception as e:
            print("Error loading model:", e)
    else:
        print(f"No saved model found at {model_path}")

# Load the model when the app starts
load_model()

@app.route('/')
def home():
    return jsonify({
        "message": "ML Model API",
        "endpoints": {
            "/predict": "POST - Make predictions",
            "/health": "GET - Check API health",
            "/metrics": "GET - Get model metrics"
        }
    })

@app.route('/health')
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None
    })

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    try:
        data = request.get_json()
        input_value = data.get('input')
        if input_value is None:
            return jsonify({"error": "No input provided"}), 400

        X = np.array([[input_value]], dtype=np.float32)
        prediction = model.predict(X, verbose=0)
        return jsonify({
            "input": float(input_value),
            "prediction": float(prediction[0][0])
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/metrics')
def metrics():
    if metrics_text:
        return jsonify({"metrics": metrics_text})
    else:
        return jsonify({"error": "Metrics not available"}), 404

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
