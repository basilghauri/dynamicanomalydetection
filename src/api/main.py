"""
FastAPI server for dynamic anomaly detection inference
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any, Union
import numpy as np
import logging
from datetime import datetime
import os
from pathlib import Path

from src.models.dynamic_anomaly_detector import DynamicAnomalyDetector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Dynamic Anomaly Detection API",
    description="Real-time anomaly detection for transaction data with dynamic feature support",
    version="1.0.0"
)

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instance
model: Optional[DynamicAnomalyDetector] = None
MODEL_PATH = os.getenv("MODEL_PATH", "models/anomaly_detector.joblib")


# Pydantic models for request/response
class TransactionData(BaseModel):
    """Single transaction data point"""
    data: Dict[str, Union[float, int, str]] = Field(
        ...,
        description="Transaction attributes as key-value pairs"
    )


class BatchTransactionData(BaseModel):
    """Batch of transaction data points"""
    transactions: List[Dict[str, Union[float, int, str]]] = Field(
        ...,
        description="List of transaction dictionaries"
    )


class AnomalyPrediction(BaseModel):
    """Anomaly prediction response"""
    is_anomaly: bool = Field(..., description="True if transaction is anomalous")
    anomaly_score: float = Field(..., description="Anomaly score (higher = more anomalous)")
    prediction: int = Field(..., description="Raw prediction (-1 for anomaly, 1 for normal)")
    timestamp: str = Field(..., description="Prediction timestamp")


class BatchAnomalyResponse(BaseModel):
    """Batch prediction response"""
    predictions: List[AnomalyPrediction]
    total_transactions: int
    anomalies_detected: int
    timestamp: str


class ModelInfoResponse(BaseModel):
    """Model information response"""
    is_fitted: bool
    num_features: int
    features: List[str]
    contamination: float
    n_estimators: int
    feature_stats: Dict[str, Dict[str, float]]


@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    global model
    try:
        model = DynamicAnomalyDetector()
        model_path = Path(MODEL_PATH)
        if model_path.exists():
            model.load(str(model_path))
            logger.info(f"Model loaded from {MODEL_PATH}")
        else:
            logger.warning(f"Model file not found at {MODEL_PATH}. Model will need to be trained.")
            model = DynamicAnomalyDetector()
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        model = DynamicAnomalyDetector()


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Dynamic Anomaly Detection API",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None and model.is_fitted,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/model/info", response_model=ModelInfoResponse)
async def get_model_info():
    """Get information about the current model"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
    
    info = model.get_model_info()
    return ModelInfoResponse(**info)


@app.post("/predict", response_model=AnomalyPrediction)
async def predict_anomaly(transaction: TransactionData):
    """
    Predict anomaly for a single transaction.
    
    The model automatically handles:
    - New features that weren't seen during training
    - Missing features (uses default values)
    - Feature scaling and normalization
    """
    if model is None or not model.is_fitted:
        raise HTTPException(
            status_code=503,
            detail="Model not fitted. Please train the model first."
        )
    
    try:
        # Predict
        predictions, scores = model.predict(transaction.data, return_scores=True)
        
        # Get first prediction (single transaction)
        prediction = predictions[0]
        score = float(scores[0])
        
        return AnomalyPrediction(
            is_anomaly=(prediction == -1),
            anomaly_score=score,
            prediction=int(prediction),
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchAnomalyResponse)
async def predict_batch(transactions: BatchTransactionData):
    """
    Predict anomalies for a batch of transactions.
    
    More efficient for processing multiple transactions at once.
    """
    if model is None or not model.is_fitted:
        raise HTTPException(
            status_code=503,
            detail="Model not fitted. Please train the model first."
        )
    
    try:
        # Predict
        predictions, scores = model.predict(transactions.transactions, return_scores=True)
        
        # Format responses
        prediction_list = []
        anomalies_count = 0
        
        for pred, score in zip(predictions, scores):
            is_anomaly = (pred == -1)
            if is_anomaly:
                anomalies_count += 1
            
            prediction_list.append(
                AnomalyPrediction(
                    is_anomaly=is_anomaly,
                    anomaly_score=float(score),
                    prediction=int(pred),
                    timestamp=datetime.now().isoformat()
                )
            )
        
        return BatchAnomalyResponse(
            predictions=prediction_list,
            total_transactions=len(transactions.transactions),
            anomalies_detected=anomalies_count,
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/train")
async def train_model(
    training_data: BatchTransactionData,
    contamination: float = 0.1,
    n_estimators: int = 100
):
    """
    Train the anomaly detection model on provided data.
    
    This endpoint allows you to train/retrain the model with new data.
    The model will automatically adapt to the features in the training data.
    """
    global model
    
    try:
        if contamination < 0 or contamination > 0.5:
            raise HTTPException(
                status_code=400,
                detail="Contamination must be between 0 and 0.5"
            )
        
        # Initialize or update model
        if model is None:
            model = DynamicAnomalyDetector(
                contamination=contamination,
                n_estimators=n_estimators
            )
        else:
            model.contamination = contamination
            model.n_estimators = n_estimators
        
        # Train model
        model.fit(training_data.transactions)
        
        # Save model
        model_path = Path(MODEL_PATH)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(model_path))
        
        return {
            "message": "Model trained successfully",
            "num_samples": len(training_data.transactions),
            "num_features": len(model.known_features),
            "features": model.known_features,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Training error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/update")
async def update_model(training_data: BatchTransactionData):
    """
    Incrementally update the model with new data.
    
    Note: This refits the model. For true incremental learning,
    you may need to implement a data buffer.
    """
    global model
    
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not initialized. Please train the model first."
        )
    
    try:
        # Partial fit (currently refits, can be enhanced for true incremental learning)
        model.partial_fit(training_data.transactions)
        
        # Save updated model
        model_path = Path(MODEL_PATH)
        model.save(str(model_path))
        
        return {
            "message": "Model updated successfully",
            "num_new_samples": len(training_data.transactions),
            "total_features": len(model.known_features),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Update error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

