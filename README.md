# Agentic Self-RAG: Multi-Agent Self-Correcting RAG

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)]()

**Publication-Ready Implementation** of Agentic Self-RAG with complete evaluation, baselines, statistical testing, and streaming batch evaluation capabilities.

---

## Architecture Diagram

<p align="center"> <img src="./architecture.png" width="100%" alt="Syed Bilal Hassan Banner"> </p>

<h1 align="center">Proposed Architecture</h1>

## 🚀 Quick Start (2 min setup)

### Automated Setup (Recommended)
```bash
# Clone and setup
git clone https://github.com/Syed-Bilal-Hassan-SBH/Agentic-Self-RAG.git && cd Agentic-Self-Rag

# Make script executable and run
chmod +x scripts/setup_env.sh
bash scripts/setup_env.sh

# This handles ALL setup:
# ✓ Python virtual environment
# ✓ All dependencies (langchain, groq, etc.)
# ✓ Spacy NLP models
# ✓ HotpotQA dataset download
# ✓ .env file creation
# ✓ Validation smoke test
```

### Manual Installation
```bash
# Virtual environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
python3 -m spacy download en_core_web_sm
python3 scripts/download_hotpotqa.py

# Setup API key
cp .env.example .env
# Edit .env and add: GROQ_API_KEY=your_api_key_here (get from https://console.groq.com)
```

### Verify Installation
```bash
# Run smoke test (5 samples)
NUM_SAMPLES=5 python experiments/run_main_evaluation.py
```

---

## 📊 Running Experiments

### Main Evaluation (500 samples)
```bash
python experiments/run_main_evaluation.py
```

**What it does:**
1. Loads HotpotQA validation set
2. Initializes 4 systems (Vanilla RAG, Self-RAG variants, Agentic Self-RAG)
3. Evaluates each on 500 questions
4. Computes EM/F1 metrics
5. Performs statistical significance tests
6. Generates results (CSV, JSON)

**Output:** `results/main_evaluation/`
- `results_500.csv` - Per-sample predictions
- `summary.json` - Aggregate metrics
- `statistical_comparisons.json` - Significance tests
- `metadata.json` - Experiment configuration

**Expected Time:** ~30 minutes
**Expected EM Results:** 45-54% across systems

---

### Streaming Batch Evaluation (Recommended)
```bash
python run_streaming_evaluation.py
```

**Real-time evaluation with progressive output:**
- Processes 500 samples in batches of 50
- Generates side-by-side CSV with all systems in one row
- Real-time log file updates as samples complete
- Multiple output formats for analysis

**Output:** `results/batch_evaluation/`
- `sidebyside_results_500.csv` - All systems visible per row (Excel-friendly)
- `detailed_results_500.csv` - Traditional format for analysis
- `streaming_log_500.txt` - Human-readable real-time logs
- `streaming_summary_500.json` - Final aggregated statistics

**Expected Time:** ~45 minutes

---

### Baseline Comparison (Advanced Metrics)
```bash
python experiments/compare_baselines.py
```

**Computes:**
- Standard metrics (EM, F1)
- Advanced metrics (ROUGE-L, BERTScore)
- Hallucination metrics (Context Overlap, Faithfulness)
- Entity-level hallucination detection
- Semantic entropy
- Latency benchmarks
- Error analysis

**Output:** `results/baselines/`
- `baseline_comparison_summary.json` - All metrics
- `baseline_comparison.csv` - Formatted comparison table
- `pairwise_comparisons.json` - Statistical tests

---

### Generate Paper Tables & Figures
```bash
python experiments/generate_paper_tables.py
```

**Produces:**
- LaTeX tables for publication
- Markdown summary report
- PDF comparison plots
- Statistical significance tables
- Error analysis breakdown

**Output:** `results/paper_tables/`

---

## 🧪 Running Tests

```bash
# All tests
pytest tests/ -v

# Specific test class
pytest tests/test_integration.py::TestMetrics -v

# With coverage
pytest tests/ --cov=src --cov-report=html
```

**Test Coverage:**
- ✓ 15+ metric computation tests
- ✓ 10+ statistical test functions
- ✓ Data loading and reproducibility
- ✓ System initialization
- ✓ Complete evaluation pipeline

---

## 📁 Project Structure

```
agentic-self-rag/
├── src/
│   ├── agentic_self_rag.py         # Main multi-agent system
│   ├── enhanced_agentic_self_rag.py # Enhanced version
│   ├── ultra_agentic_rag.py         # Ultra version
│   ├── improved_agents.py           # Improved agent implementations
│   ├── vanilla_rag.py               # Simple RAG baseline
│   ├── self_rag.py                  # Self-RAG variants
│   ├── config.py                    # Configuration
│   │
│   ├── agents/                      # Multi-agent modules
│   │   ├── query_analyzer_agent.py
│   │   ├── retrieval_critic_agent.py
│   │   ├── answer_verifier_agent.py
│   │   └── multi_hop_agent.py
│   │
│   ├── baselines/                   # Baseline implementations
│   │   ├── published_self_rag.py
│   │   └── react_agent.py
│   │
│   ├── evaluation/                  # Metrics & testing
│   │   ├── metrics.py               # EM, F1, hallucination detection
│   │   ├── statistical_tests.py     # Significance testing
│   │   └── error_analysis.py        # Failure taxonomy
│   │
│   └── utils/                       # Utilities
│       ├── data_utils.py            # HotpotQA loading
│       ├── logging_utils.py         # Structured logging
│       ├── retriever.py             # Retrieval interface
│       └── repro.py                 # Reproducibility
│
├── experiments/
│   ├── run_main_evaluation.py       # Main evaluation script
│   ├── run_batch_evaluation.py      # Batch evaluation with side-by-side output
│   ├── run_batch_evaluation_streaming.py  # Streaming batch evaluation
│   ├── compare_baselines.py         # Baseline comparison
│   ├── generate_paper_tables.py     # Publication tables
│   ├── run_error_analysis.py        # Error analysis
│   └── run_ablations.py             # Ablation studies
│
├── tests/
│   ├── test_integration.py          # Integration tests
│   ├── test_metrics.py              # Metric tests
│   ├── test_verifier.py             # Verifier tests
│   └── test_reproducibility.py      # Reproducibility tests
│
├── scripts/
│   ├── setup_env.sh                 # Automated setup
│   ├── download_hotpotqa.py         # Dataset download
│   ├── organize_results.py          # Result analysis and formatting
│   ├── run_full_evaluation.sh       # Full evaluation script
│   └── run_smoke_test.sh            # Quick smoke test
│
├── data/                            # Datasets (gitignored)
│   ├── hotpotqa_sample.json
│   └── hotpotqa_cache.json
│
├── results/                         # Experiment outputs
│   ├── main_evaluation/            # Main evaluation results
│   ├── batch_evaluation/           # Streaming batch evaluation results
│   ├── baselines/                  # Baseline comparison results
│   ├── paper_tables/               # Publication-ready tables
│   ├── error_analysis/             # Error analysis results
│   └── *.csv, *.json, *.log        # Various result files
│
├── configs/
│   └── experiment_config.yaml       # Experiment configuration
│
├── requirements.txt                 # Python dependencies
├── Dockerfile                       # Container image
├── docker-compose.yml               # Multi-service setup
├── Makefile                         # Common commands
├── DOCKER_DEPLOYMENT_GUIDE.md      # Docker deployment instructions
├── PROJECT_STRUCTURE.txt            # Detailed project structure
├── generate_all_tables.py          # Table generation utility
├── generate_visual_tables.py       # Visual table generation
├── debug_metrics.py                # Debugging utilities
└── README.md                        # This file
```

---

## 🔧 Configuration

Edit `configs/experiment_config.yaml`:

```yaml
experiment:
  random_seed: 42
  num_samples: 500
  stratify_by_type: true

dataset:
  split: validation
  cache_dir: data/

agentic_rag:
  max_iterations: 4
  confidence_threshold_high: 0.85
  confidence_threshold_low: 0.50
  use_multi_hop: true
  use_adaptive_iteration: true

retrieval:
  k: 5
  multi_hop_k: 10
  embedding_model: "sentence-transformers/all-MiniLM-L6-v2"

model:
  name: "llama-3.3-70b-versatile"
  temperature: 0.0
  max_tokens: 1024
```

---

## 📊 Evaluation Metrics

### Core Metrics
- **EM (Exact Match)**: Binary match after normalization
- **F1**: Token-level overlap with precision/recall

### Advanced Metrics
- **ROUGE-L**: Longest common subsequence
- **BERTScore**: Semantic similarity
- **Hallucination Detection**:
  - Context overlap ratio
  - Faithfulness score
  - Entity-level hallucination rate
- **Semantic Entropy**: Uncertainty in predictions
- **Refusal Metrics**: Answer vs abstention rate

### Statistical Tests
- **Paired t-test**: Assumes normality
- **Wilcoxon test**: Non-parametric alternative
- **Cohen's d**: Effect size (small/medium/large)
- **Bootstrap CI**: Confidence intervals

---

## 📈 Expected Results

### 500-Sample Evaluation
```
System                    EM         F1         Latency
────────────────────────────────────────────────────────
Vanilla RAG               45.2%      0.612      2.34s
Simplified Self-RAG       49.1%      0.638      3.12s
Published Self-RAG        52.8%      0.658      3.85s
Agentic Self-RAG          54.3% ↑    0.671 ↑    4.21s

Statistical Significance (Agentic vs Vanilla):
  Δ EM: +9.1% (p < 0.001) ***
  Δ F1: +0.059 (p < 0.001) ***
  Effect Size (Cohen's d): 0.251
```

### Per-Type Performance
```
Question Type         Vanilla RAG      Agentic Self-RAG    Improvement
─────────────────────────────────────────────────────────────────────────
Bridge (2-hop)        42.0% EM          51.2% EM           +9.2%
Comparison (2-hop)    48.4% EM          57.4% EM           +9.0%
```

### System Variants
The project includes multiple RAG system implementations:
- **Vanilla RAG**: Simple retrieval-augmented generation baseline
- **Simplified Self-RAG**: Basic self-reflection mechanism
- **Published Self-RAG**: Implementation based on published Self-RAG paper
- **Agentic Self-RAG**: Multi-agent system with query analysis, retrieval criticism, and answer verification
- **Enhanced Agentic Self-RAG**: Improved version with enhanced agents
- **Ultra Agentic RAG**: Advanced multi-hop reasoning capabilities

---

## 🐳 Docker Usage

### Build Image
```bash
docker build -t agentic-self-rag .
```

### Run Smoke Test
```bash
docker run --rm \
  -v $(pwd)/results:/app/results \
  -e GROQ_API_KEY=$GROQ_API_KEY \
  -e NUM_SAMPLES=5 \
  agentic-self-rag
```

### Run with Docker Compose
```bash
# Full evaluation
docker-compose up evaluation

# Baseline comparison
docker-compose up baseline-comparison

# Generate tables
docker-compose up paper-tables
```

---

## 🔍 Troubleshooting

### API Key Issues
```bash
# Check key is set
echo $GROQ_API_KEY

# Add to .env if missing
echo "GROQ_API_KEY=your_key" >> .env
```

### Memory Issues
```bash
# Reduce sample size
NUM_SAMPLES=50 python experiments/run_main_evaluation.py

# Force CPU mode
export CUDA_VISIBLE_DEVICES=""
python experiments/run_main_evaluation.py
```

### HotpotQA Download
```bash
# Force re-download
python scripts/download_hotpotqa.py --force

# Use cache
rm data/hotpotqa_*.json  # Clear cache
```

---

## ✅ Reproducibility

- ✓ Fixed random seeds (42)
- ✓ Pinned dependency versions
- ✓ Deterministic dataset loading
- ✓ Git commit recorded
- ✓ Environment info saved
- ✓ Docker environment included

### Verify Reproducibility
```bash
python -c "
from src.utils.data_utils import load_hotpotqa
from src.utils.repro import set_seed

set_seed(42)
data1 = load_hotpotqa(split='validation', num_samples=10)
data2 = load_hotpotqa(split='validation', num_samples=10)

assert data1[0]['question'] == data2[0]['question']
print('✓ Reproducibility verified')
"
```

---

## 📚 Citation

```bibtex
@inproceedings{hassan2025agentic,
  title={Agentic Self-RAG: Multi-Agent Reasoning for Self-Correcting Retrieval-Augmented Generation},
  author={Hassan, Syed Bilal and Abdullah and Abbas, Mohsin},
  booktitle={EMNLP 2025},
  year={2025}
}
```

---

## 📧 Contact

For questions or issues:
- Email: syedbilal8803@gmail.com

---


## 🙏 Acknowledgments

- HotpotQA dataset creators
- LangChain and LangGraph teams
- Groq API for free LLM access
- FAIR team for Self-RAG inspiration
