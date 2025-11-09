"""
Example usage of the dynamic anomaly detection model
"""

import pandas as pd
import requests
import json
from src.models.dynamic_anomaly_detector import DynamicAnomalyDetector
from src.utils.data_generator import generate_transaction_data, generate_dynamic_transaction_data


def example_local_usage():
    """Example of using the model locally"""
    print("=" * 60)
    print("Example 1: Local Model Usage")
    print("=" * 60)
    
    # Initialize model
    model = DynamicAnomalyDetector(contamination=0.1, n_estimators=100)
    
    # Generate training data with 10 features
    print("\n1. Training model with 10 features...")
    train_data = generate_transaction_data(n_samples=500, n_features=10)
    model.fit(train_data)
    print(f"   Model trained with {len(model.known_features)} features")
    print(f"   Features: {model.known_features[:5]}...")
    
    # Test with same features
    print("\n2. Testing with same features...")
    test_data = generate_transaction_data(n_samples=100, n_features=10)
    predictions, scores = model.predict(test_data, return_scores=True)
    anomalies = sum(predictions == -1)
    print(f"   Detected {anomalies} anomalies out of {len(predictions)} transactions")
    
    # Test with NEW features (dynamic adaptation)
    print("\n3. Testing with NEW features (5 additional features)...")
    new_features = ['new_feature_1', 'new_feature_2', 'new_feature_3', 
                   'new_feature_4', 'new_feature_5']
    extended_data = generate_dynamic_transaction_data(
        base_features=train_data.columns.tolist(),
        new_features=new_features,
        n_samples=100
    )
    predictions, scores = model.predict(extended_data, return_scores=True)
    anomalies = sum(predictions == -1)
    print(f"   Model adapted to {len(model.known_features)} features")
    print(f"   Detected {anomalies} anomalies out of {len(predictions)} transactions")
    
    # Test with missing features
    print("\n4. Testing with missing features...")
    partial_data = test_data.iloc[:50, :7]  # Only first 7 features
    predictions, scores = model.predict(partial_data, return_scores=True)
    print(f"   Model handled missing features gracefully")
    print(f"   Detected {sum(predictions == -1)} anomalies")
    
    # Save model
    print("\n5. Saving model...")
    model.save("models/example_model.joblib")
    print("   Model saved successfully")
    
    print("\n" + "=" * 60)


def example_api_usage():
    """Example of using the API"""
    print("=" * 60)
    print("Example 2: API Usage")
    print("=" * 60)
    
    base_url = "http://localhost:8000"
    
    # Check health
    print("\n1. Checking API health...")
    try:
        response = requests.get(f"{base_url}/health")
        print(f"   Status: {response.json()}")
    except requests.exceptions.ConnectionError:
        print("   ERROR: API not running. Start it with: uvicorn src.api.main:app")
        return
    
    # Train model via API
    print("\n2. Training model via API...")
    train_data = generate_transaction_data(n_samples=500, n_features=10)
    train_payload = {
        "transactions": train_data.to_dict('records')
    }
    
    response = requests.post(
        f"{base_url}/train",
        json=train_payload,
        params={"contamination": 0.1, "n_estimators": 100}
    )
    print(f"   Training response: {response.json()}")
    
    # Get model info
    print("\n3. Getting model info...")
    response = requests.get(f"{base_url}/model/info")
    info = response.json()
    print(f"   Features: {info['num_features']}")
    print(f"   Is fitted: {info['is_fitted']}")
    
    # Single prediction
    print("\n4. Making single prediction...")
    transaction = train_data.iloc[0].to_dict()
    response = requests.post(
        f"{base_url}/predict",
        json={"data": transaction}
    )
    result = response.json()
    print(f"   Is anomaly: {result['is_anomaly']}")
    print(f"   Anomaly score: {result['anomaly_score']:.4f}")
    
    # Batch prediction
    print("\n5. Making batch prediction...")
    batch_data = train_data.iloc[:10].to_dict('records')
    response = requests.post(
        f"{base_url}/predict/batch",
        json={"transactions": batch_data}
    )
    result = response.json()
    print(f"   Total transactions: {result['total_transactions']}")
    print(f"   Anomalies detected: {result['anomalies_detected']}")
    
    # Test with new features
    print("\n6. Testing with new features via API...")
    new_features = ['new_feature_1', 'new_feature_2', 'new_feature_3']
    extended_data = generate_dynamic_transaction_data(
        base_features=train_data.columns.tolist(),
        new_features=new_features,
        n_samples=10
    )
    batch_data = extended_data.to_dict('records')
    response = requests.post(
        f"{base_url}/predict/batch",
        json={"transactions": batch_data}
    )
    result = response.json()
    print(f"   Model adapted to new features automatically")
    print(f"   Anomalies detected: {result['anomalies_detected']}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Dynamic Anomaly Detection - Example Usage")
    print("=" * 60)
    
    # Run local example
    example_local_usage()
    
    # Run API example (uncomment if API is running)
    # example_api_usage()
    
    print("\nExamples completed!")

