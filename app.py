# app.py
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
import tensorflow as tf
import numpy as np
import os

app = FastAPI(title="ML Model API")

# -------------------------------
# Request body schema
# -------------------------------
class PredictRequest(BaseModel):
    input: float

# -------------------------------
# Global model, metrics, and plot
# -------------------------------
model = None
metrics_text = ""
plot_file = os.path.join(os.path.dirname(__file__), "model_results.png")

# -------------------------------
# Load trained model
# -------------------------------
def load_model():
    global model, metrics_text
    model_path = os.path.join(os.path.dirname(__file__), "model.keras")
    print("Looking for model at:", model_path)

    if os.path.exists(model_path):
        try:
            model = tf.keras.models.load_model(model_path)
            print("Model loaded successfully!")

            # Example test data
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

# Load model at startup
load_model()

# -------------------------------
# API Endpoints
# -------------------------------

@app.get("/", summary="Home")
def home():
    return {
        "message": "ML Model API",
        "endpoints": {
            "/predict": "POST - Make predictions",
            "/health": "GET - Check API health",
            "/metrics": "GET - Get model metrics",
            "/plot": "GET - Get model result plot"
        }
    }

@app.get("/health", summary="Check API Health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }

@app.get("/metrics", summary="Get Model Metrics", responses={200: {"content": {"application/json": {}}}, 404: {"description": "Metrics not available"}})
def metrics():
    if metrics_text:
        return {"metrics": metrics_text}
    else:
        raise HTTPException(status_code=404, detail="Metrics not available")

@app.get("/plot", summary="Get Model Result Plot", responses={200: {"content": {"image/png": {}}}, 404: {"description": "Plot not found"}})
def get_plot():
    if os.path.exists(plot_file):
        return FileResponse(plot_file, media_type="image/png")
    else:
        raise HTTPException(status_code=404, detail="Plot not found")

@app.post("/predict", summary="Make Predictions", responses={200: {"content": {"application/json": {}}}, 400: {"description": "Bad Request"}, 500: {"description": "Model not loaded"}})
def predict(request: PredictRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        X = np.array([[request.input]], dtype=np.float32)
        prediction = model.predict(X, verbose=0)
        return {"input": request.input, "prediction": float(prediction[0][0])}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
