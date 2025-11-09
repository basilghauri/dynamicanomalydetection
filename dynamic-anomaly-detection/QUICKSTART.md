# Quick Start Guide

## 1. Local Development Setup

```bash
# Navigate to project directory
cd dynamic-anomaly-detection

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create necessary directories
mkdir -p models data
```

## 2. Run Example Usage

```bash
# Run the example script
python example_usage.py
```

This will demonstrate:
- Training with 10 features
- Predicting with same features
- Adapting to new features (5 additional)
- Handling missing features

## 3. Start API Server

```bash
# Start the FastAPI server
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

Or use the Makefile:
```bash
make run
```

## 4. Test the API

### Health Check
```bash
curl http://localhost:8000/health
```

### Train Model
```bash
curl -X POST "http://localhost:8000/train?contamination=0.1" \
  -H "Content-Type: application/json" \
  -d @- << EOF
{
  "transactions": [
    {"amount": 100.0, "merchant_category": "retail", "hour_of_day": 14},
    {"amount": 200.0, "merchant_category": "restaurant", "hour_of_day": 19}
  ]
}
EOF
```

### Predict Single Transaction
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "data": {
      "amount": 150.0,
      "merchant_category": "retail",
      "hour_of_day": 14
    }
  }'
```

## 5. Docker Deployment

### Build and Run
```bash
# Build image
docker build -t dynamic-anomaly-detector .

# Run with docker-compose
docker-compose up -d

# Or run directly
docker run -p 8000:8000 -v $(pwd)/models:/app/models dynamic-anomaly-detector
```

### Check Status
```bash
# Check if container is running
docker ps

# Check logs
docker logs dynamic-anomaly-detector

# Health check
curl http://localhost:8000/health
```

## 6. Run Tests

```bash
# Install pytest if not already installed
pip install pytest

# Run tests
pytest tests/test_model.py -v
```

## Key Features Demonstrated

1. **Dynamic Feature Adaptation**: Train with 10 features, predict with 15 features
2. **Missing Feature Handling**: Predict with only 7 features, model uses defaults
3. **Real-time Inference**: Fast API responses (< 10ms typically)
4. **Batch Processing**: Efficient batch predictions

## Next Steps

- Customize the model parameters in `src/models/dynamic_anomaly_detector.py`
- Add your own data preprocessing in `src/utils/`
- Integrate with your transaction data pipeline
- Deploy to production using Docker/Kubernetes

