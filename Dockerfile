# Dockerfile
# Reproducible environment for Agentic Self-RAG
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download spaCy model
RUN python -m spacy download en_core_web_sm

# Copy source code
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONHASHSEED=42

# Create results directory
RUN mkdir -p results

# Default command
CMD ["python", "experiments/run_main_evaluation.py"]