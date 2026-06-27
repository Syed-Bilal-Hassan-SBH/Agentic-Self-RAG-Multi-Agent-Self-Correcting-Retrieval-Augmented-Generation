#!/bin/bash
# setup_env.sh - Complete environment setup for Agentic Self-RAG

set -e  # Exit on error

echo "================================"
echo "Agentic Self-RAG Setup"
echo "================================"
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Create virtual environment
echo ""
echo "Creating virtual environment..."
if [ -d "venv" ]; then
    echo "Virtual environment already exists. Skipping creation."
else
    python3 -m venv venv
    echo "✓ Virtual environment created"
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/bin/activate || source venv/Scripts/activate
echo "✓ Virtual environment activated"

# Upgrade pip, setuptools, wheel
echo ""
echo "Upgrading pip, setuptools, wheel..."
pip install --upgrade pip setuptools wheel
echo "✓ Package managers upgraded"

# Install core dependencies
echo ""
echo "Installing core dependencies..."
pip install -r requirements.txt
echo "✓ Core dependencies installed"

# Download spacy model
echo ""
echo "Downloading spacy language model..."
python3 -m spacy download en_core_web_sm
echo "✓ Spacy model downloaded"

# Create .env file if it doesn't exist
echo ""
echo "Setting up .env file..."
if [ -f ".env" ]; then
    echo "✓ .env file already exists"
else
    cat > .env << EOF
# Agentic Self-RAG Configuration
# Add your API keys below

# Groq API key (free tier available at https://console.groq.com)
GROQ_API_KEY=your_api_key_here

# Optional: HuggingFace token for private models
HF_TOKEN=

# Optional: Weights & Biases API key for experiment tracking
WANDB_API_KEY=
EOF
    echo "✓ .env file created (please add your API keys)"
fi

# Download HotpotQA dataset
echo ""
echo "Downloading HotpotQA dataset (this may take a few minutes)..."
python3 scripts/download_hotpotqa.py
echo "✓ HotpotQA dataset downloaded"

# Create required directories
echo ""
echo "Creating required directories..."
mkdir -p data
mkdir -p results/main_evaluation
mkdir -p results/baselines
mkdir -p results/ablations
mkdir -p results/error_analysis
mkdir -p models
mkdir -p logs
echo "✓ Directories created"

# Run smoke test
echo ""
echo "Running smoke test (5 samples)..."
echo "This will verify that the system is working correctly."
NUM_SAMPLES=5 python3 experiments/run_main_evaluation.py

echo ""
echo "================================"
echo "✓ Setup complete!"
echo "================================"
echo ""
echo "Next steps:"
echo "1. Edit .env and add your GROQ_API_KEY"
echo "2. Run: make smoke  (5-sample test)"
echo "3. Run: make full   (500-sample full evaluation)"
echo ""
echo "For more information, see README.md"
