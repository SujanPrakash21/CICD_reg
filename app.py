from flask import Flask, request, jsonify, send_file
import tensorflow as tf
import numpy as np
import os

app = Flask(__name__)

model = None
metrics_text = ""

def load_model():
    global model, metrics_text
    model_path = os.path.join(os.path.dirname(__file__), 'model.keras')
    print("Looking for model at:", model_path)
    if os.path.exists(model_path):
        try:
            model = tf.keras.models.load_model(model_path)
            print("Model loaded successfully!")

            # Optional: load metrics from file
            metrics_file = os.path.join(os.path.dirname(__file__), 'metrics.txt')
            if os.path.exists(metrics_file):
                with open(metrics_file, 'r') as f:
                    metrics_text = f.read()
        except Exception as e:
            print("Error loading model:", e)
    else:
        print("No saved model found.")

load_model()

@app.route('/')
def home():
    return jsonify({
        "message": "ML Model API",
        "endpoints": {
            "/predict": "POST",
            "/health": "GET",
            "/metrics": "GET",
            "/plot": "GET"
        }
    })

@app.route('/health')
def health():
    return jsonify({"status": "healthy", "model_loaded": model is not None})

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
        return jsonify({"input": float(input_value), "prediction": float(prediction[0][0])})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/metrics')
def metrics():
    if metrics_text:
        return jsonify({"metrics": metrics_text})
    else:
        return jsonify({"error": "Metrics not available"}), 404

@app.route('/plot')
def plot():
    plot_path = os.path.join(os.path.dirname(__file__), 'model_results.png')
    if os.path.exists(plot_path):
        return send_file(plot_path, mimetype='image/png')
    else:
        return jsonify({"error": "Plot not available"}), 404

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
