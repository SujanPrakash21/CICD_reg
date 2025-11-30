# app.py
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import FileResponse
from pydantic import BaseModel
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt

app = FastAPI(title="ML Model API with Plot & Metrics")

# -----------------------
# Request Schema
# -----------------------
class PredictRequest(BaseModel):
    input: float

# -----------------------
# Global Variables
# -----------------------
model = None
metrics_text = ""

# -----------------------
# Load Model + Metrics
# -----------------------
def load_model_and_prepare():
    global model, metrics_text

    model_path = os.path.join(os.path.dirname(__file__), "model.keras")
    print("Looking for model at:", model_path)

    if not os.path.exists(model_path):
        print("Model not found!")
        return

    try:
        model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully!")

        # Same data used during training
        X = np.arange(-100, 100, 4).reshape(-1, 1)
        y = np.arange(-90, 110, 4).reshape(-1, 1)

        X_train = X[:40]
        y_train = y[:40]
        X_test = X[40:]
        y_test = y[40:]

        # Predictions
        y_preds = model.predict(X_test, verbose=0)

        # Metrics
        mae_fn = tf.keras.losses.MeanAbsoluteError()
        mse_fn = tf.keras.losses.MeanSquaredError()

        mae_val = float(mae_fn(y_test, y_preds).numpy())
        mse_val = float(mse_fn(y_test, y_preds).numpy())

        metrics_text = (
            f"Mean Absolute Error = {mae_val:.2f}, "
            f"Mean Squared Error = {mse_val:.2f}"
        )

    except Exception as e:
        print("Error loading model:", e)

# Load model on startup
load_model_and_prepare()

# -----------------------
# Home
# -----------------------
@app.get("/")
def home():
    return {
        "message": "ML Model API",
        "endpoints": {
            "/predict": "POST - Make predictions",
            "/health": "GET - Check API health",
            "/metrics": "GET - Get model metrics",
            "/plot": "GET - View model result plot"
        }
    }

# -----------------------
# Health Check
# -----------------------
@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None}

# -----------------------
# Metrics
# -----------------------
@app.get("/metrics")
def metrics():
    if metrics_text:
        return {"metrics": metrics_text}
    raise HTTPException(status_code=404, detail="Metrics not available")

# -----------------------
# Dynamic Plot (Swagger supported)
# -----------------------
@app.get("/plot")
def get_plot():
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    # Recreate training/testing data
    X = np.arange(-100, 100, 4).reshape(-1, 1)
    y = np.arange(-90, 110, 4).reshape(-1, 1)

    X_train = X[:40]
    y_train = y[:40]
    X_test = X[40:]
    y_test = y[40:]

    # Predictions
    y_preds = model.predict(X_test, verbose=0)

    # ---- In-memory plot ----
    import io
    fig, ax = plt.subplots(figsize=(6, 5))

    ax.scatter(X_train, y_train, c="b", label="Training data")
    ax.scatter(X_test, y_test, c="g", label="Testing data")
    ax.scatter(X_test, y_preds, c="r", label="Predictions")

    ax.legend()
    ax.set_title("Model Results")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=120)
    buf.seek(0)
    plt.close()

    return Response(content=buf.getvalue(), media_type="image/png")

# -----------------------
# Prediction Endpoint
# -----------------------
@app.post("/predict")
def predict(request: PredictRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    X = np.array([[request.input]], dtype=np.float32)
    prediction = model.predict(X, verbose=0)

    return {"input": request.input, "prediction": float(prediction[0][0])}
