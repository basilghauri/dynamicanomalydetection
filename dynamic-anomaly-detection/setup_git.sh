#!/bin/bash

# Setup script for initializing git and preparing for GitHub

echo "🚀 Setting up Git repository for Dynamic Anomaly Detection"

# Check if git is available
if ! command -v git &> /dev/null; then
    echo "❌ Git is not installed. Please install Git first."
    exit 1
fi

# Initialize git repository if not already initialized
if [ ! -d .git ]; then
    echo "📦 Initializing Git repository..."
    git init
else
    echo "✅ Git repository already initialized"
fi

# Add all files
echo "📝 Adding files to Git..."
git add .

# Check if there are changes to commit
if git diff --staged --quiet; then
    echo "ℹ️  No changes to commit"
else
    echo "💾 Committing changes..."
    git commit -m "Initial commit: Dynamic Anomaly Detection System

- Production-ready anomaly detection with dynamic feature adaptation
- FastAPI REST API for real-time inference
- Docker support for containerized deployment
- Comprehensive documentation and examples
- Isolation Forest-based ML model
- Automatic handling of new and missing features"
    
    echo "✅ Changes committed successfully!"
    echo ""
    echo "📋 Next steps to push to GitHub:"
    echo "1. Create a new repository on GitHub (https://github.com/new)"
    echo "2. Run the following commands:"
    echo ""
    echo "   git remote add origin https://github.com/YOUR_USERNAME/dynamic-anomaly-detection.git"
    echo "   git branch -M main"
    echo "   git push -u origin main"
    echo ""
fi

echo ""
echo "✨ Setup complete!"

