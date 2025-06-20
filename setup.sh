#!/bin/bash

# Political Affiliation Survey - Setup Script
echo "Setting up Political Affiliation Survey ML Environment..."

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is required but not installed. Please install Python 3 first."
    exit 1
fi

# Create virtual environment (optional but recommended)
echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install required packages
echo "Installing required packages..."
pip install -r requirements.txt

echo "Setup complete!"
echo ""
echo "To use the survey system:"
echo "1. Collect survey data by running: java -cp java Main"
echo "2. Train the ML model: python python/train.py"
echo "3. Make predictions: python python/predict.py"
echo ""
echo "To activate the virtual environment in the future:"
echo "source venv/bin/activate"
