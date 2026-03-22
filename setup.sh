#!/bin/bash

echo "🚀 Setting up Dynamic Trend & Event Detector..."

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

echo "✅ Environment setup complete."
echo "To run the analysis:"
echo "source venv/bin/activate"
echo "python src/eda.py"
echo "python src/baseline.py"
echo "python src/advanced_ml.py"
