#!/bin/bash

# Quick start script for the dynamic anomaly detection system

echo "=========================================="
echo "Dynamic Anomaly Detection - Quick Start"
echo "=========================================="
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo "Creating directories..."
mkdir -p models data

# Run example
echo ""
echo "Running example usage..."
echo ""
python example_usage.py

echo ""
echo "=========================================="
echo "To start the API server, run:"
echo "  uvicorn src.api.main:app --host 0.0.0.0 --port 8000"
echo "=========================================="

