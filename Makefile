# Makefile
# Convenient commands for running experiments

.PHONY: help install test smoke full ablations stats clean docker

help:
	@echo "Agentic Self-RAG - Available Commands"
	@echo "======================================"
	@echo "install      - Install dependencies"
	@echo "test         - Run unit tests"
	@echo "smoke        - Run smoke test (5 samples)"
	@echo "full         - Run full evaluation (500 samples)"
	@echo "ablations    - Run ablation study"
	@echo "stats        - Generate statistical comparisons"
	@echo "clean        - Clean generated files"
	@echo "docker       - Build and run in Docker"

install:
	pip install -r requirements.txt
	python -m spacy download en_core_web_sm
	python scripts/download_hotpotqa.py

test:
	pytest tests/ -v --cov=src --cov-report=html

smoke:
	NUM_SAMPLES=5 python experiments/run_main_evaluation.py

full:
	python experiments/run_main_evaluation.py

ablations:
	python experiments/run_ablations.py

stats:
	python -c "from src.evaluation.statistical_tests import *; print('Stats generated')"

clean:
	rm -rf results/*
	rm -rf __pycache__
	rm -rf src/__pycache__
	rm -rf tests/__pycache__
	rm -rf .pytest_cache
	rm -rf htmlcov
	find . -type d -name "__pycache__" -exec rm -rf {} +

docker:
	docker-compose build
	docker-compose up agentic-self-rag