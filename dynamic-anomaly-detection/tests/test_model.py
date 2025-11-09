"""
Tests for the dynamic anomaly detection model
"""

import pytest
import numpy as np
import pandas as pd
from src.models.dynamic_anomaly_detector import DynamicAnomalyDetector
from src.utils.data_generator import generate_transaction_data, generate_dynamic_transaction_data


def test_model_initialization():
    """Test model initialization"""
    model = DynamicAnomalyDetector(contamination=0.1, n_estimators=50)
    assert model.contamination == 0.1
    assert model.n_estimators == 50
    assert not model.is_fitted


def test_model_fit():
    """Test model fitting"""
    model = DynamicAnomalyDetector(contamination=0.1)
    data = generate_transaction_data(n_samples=100, n_features=10)
    
    model.fit(data)
    assert model.is_fitted
    assert len(model.known_features) > 0


def test_model_prediction():
    """Test model prediction"""
    model = DynamicAnomalyDetector(contamination=0.1)
    train_data = generate_transaction_data(n_samples=200, n_features=10)
    test_data = generate_transaction_data(n_samples=50, n_features=10)
    
    model.fit(train_data)
    predictions, scores = model.predict(test_data, return_scores=True)
    
    assert len(predictions) == 50
    assert len(scores) == 50
    assert all(pred in [-1, 1] for pred in predictions)


def test_dynamic_features():
    """Test model with dynamically changing features"""
    model = DynamicAnomalyDetector(contamination=0.1)
    
    # Train with initial features
    initial_data = generate_transaction_data(n_samples=200, n_features=10)
    model.fit(initial_data)
    initial_features = len(model.known_features)
    
    # Predict with new features added
    new_features = ['new_feature_1', 'new_feature_2', 'new_feature_3']
    extended_data = generate_dynamic_transaction_data(
        base_features=initial_data.columns.tolist(),
        new_features=new_features,
        n_samples=50
    )
    
    predictions, scores = model.predict(extended_data, return_scores=True)
    
    # Model should handle new features
    assert len(predictions) == 50
    assert len(model.known_features) >= initial_features


def test_missing_features():
    """Test model with missing features"""
    model = DynamicAnomalyDetector(contamination=0.1)
    
    # Train with full feature set
    train_data = generate_transaction_data(n_samples=200, n_features=10)
    model.fit(train_data)
    
    # Predict with subset of features
    test_data = train_data.iloc[:10, :5]  # Only first 5 features
    
    predictions, scores = model.predict(test_data, return_scores=True)
    assert len(predictions) == 10


def test_save_load():
    """Test model save and load"""
    model = DynamicAnomalyDetector(contamination=0.1)
    data = generate_transaction_data(n_samples=200, n_features=10)
    model.fit(data)
    
    # Save
    model.save("test_model.joblib")
    
    # Load
    new_model = DynamicAnomalyDetector()
    new_model.load("test_model.joblib")
    
    assert new_model.is_fitted
    assert len(new_model.known_features) == len(model.known_features)
    
    # Clean up
    import os
    os.remove("test_model.joblib")


def test_single_transaction_prediction():
    """Test prediction on single transaction dict"""
    model = DynamicAnomalyDetector(contamination=0.1)
    train_data = generate_transaction_data(n_samples=200, n_features=10)
    model.fit(train_data)
    
    # Single transaction as dict
    transaction = train_data.iloc[0].to_dict()
    predictions, scores = model.predict(transaction, return_scores=True)
    
    assert len(predictions) == 1
    assert len(scores) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

