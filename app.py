# app.py - Flask API to serve your model predictions

from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import os

app = Flask(__name__)

# Load the trained model (you'll need to save it first)
model = None

def load_model():
    global model
    if os.path.exists('saved_model'):
        model_path = os.path.join(os.path.dirname(__file__), 'saved_model')
        print("Model loaded successfully!")
    else:
        print("No saved model found. Train the model first.")

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
        
        # Prepare input
        X = np.array([[input_value]], dtype=np.float32)
        
        # Make prediction
        prediction = model.predict(X, verbose=0)
        
        return jsonify({
            "input": float(input_value),
            "prediction": float(prediction[0][0])
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/metrics')
def metrics():
    try:
        with open('metrics.txt', 'r') as f:
            metrics_text = f.read()
        return jsonify({
            "metrics": metrics_text
        })
    except FileNotFoundError:
        return jsonify({
            "error": "Metrics file not found"
        }), 404

if __name__ == '__main__':
    load_model()
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
