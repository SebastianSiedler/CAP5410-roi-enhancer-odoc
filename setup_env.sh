#!/bin/bash
# Setup script for baseline U-Net training environment

echo "=========================================="
echo "Setting up Baseline U-Net Environment"
echo "=========================================="
echo ""

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt

# Install additional dependencies if needed
echo "Installing additional packages (pandas, tqdm)..."
pip install pandas tqdm

echo ""
echo "=========================================="
echo "Environment setup complete!"
echo "=========================================="
echo ""
echo "To activate the environment manually, run:"
echo "  source .venv/bin/activate"
echo ""
echo "To test the baseline implementation, run:"
echo "  python test_baseline.py"
echo ""
echo "To start training, run:"
echo "  cd src && python main.py --epochs 100 --batch_size 8"
echo ""
