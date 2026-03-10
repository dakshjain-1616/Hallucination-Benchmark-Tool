# Hallucination Benchmark Tool

[![Powered by](https://img.shields.io/badge/made%20by-NEO-black)](https://heyneo.so/)

> **Built autonomously by [NEO](https://heyneo.so/) — Your Autonomous AI Agent.**

A production-ready tool for evaluating Large Language Model (LLM) responses for factual accuracy, fabricated citations, and unsupported claims.

## Features

- **Comprehensive Metrics**: Faithfulness, factual consistency, hallucination rate, BLEU score
- **LLM-as-Judge**: Semantic evaluation using OpenRouter API
- **Rule-Based Checks**: Pattern matching for fabricated citations and claim extraction
- **Modular Architecture**: Easy to swap evaluation metrics and datasets
- **Multiple Input Formats**: JSON, JSONL, CSV, Parquet
- **Detailed Reporting**: JSON, CSV, and Markdown reports with per-sample scores and aggregate statistics

## Benchmark Results

Live 3-model comparison on the sample dataset — judge: `anthropic/claude-opus-4-6` (March 2026).

### Overall Comparison

| Model | Hallucination Rate | Faithfulness | Factual Consistency | BLEU | Hallucinated | Cost |
|-------|--------------------|--------------|---------------------|------|--------------|------|
| **GPT-5.4** | **13.3%** | **81.9%** | **77.8%** | 0.114 | 1 / 3 | $0.0011 |
| Gemini 3 Flash Preview | 15.0% | 71.8% | 75.0% | **0.362** | 1 / 3 | **$0.0001** |
| Claude Sonnet 4.6 | 46.7% | 12.9% | 22.2% | 0.075 | 3 / 3 | $0.0051 |

### Per-Sample Breakdown

**What is the capital of France?**

| Model | Response | Hallucinated | Judge Score |
|-------|----------|-------------|-------------|
| GPT-5.4 | `Paris.` | No | 1.00 |
| Gemini 3 Flash | `The capital of France is **Paris**.` | No | 1.00 |
| Claude Sonnet 4.6 | `**Paris** is the capital... [+ political/cultural context]` | Yes | 0.60 |

**Who wrote 'Romeo and Juliet'?**

| Model | Response | Hallucinated | Judge Score |
|-------|----------|-------------|-------------|
| GPT-5.4 | `William Shakespeare.` | No | 1.00 |
| Gemini 3 Flash | `William Shakespeare` | No | 1.00 |
| Claude Sonnet 4.6 | `**William Shakespeare**... [+ date, plot, themes]` | Yes | 0.60 |

**What is the speed of light?**

| Model | Response | Hallucinated | Judge Score |
|-------|----------|-------------|-------------|
| GPT-5.4 | `299,792,458 m/s [+ common approximations]` | Yes | 0.75 |
| Gemini 3 Flash | `299,792,458 m/s [+ approximations]` | Yes | 0.65 |
| Claude Sonnet 4.6 | `299,792,458 m/s [+ extensive physics context]` | Yes | 0.55 |

### Key Takeaways

- **GPT-5.4** scored best overall — concise answers minimise contextual hallucination flags
- **Gemini 3 Flash** is the best value — near-identical accuracy at ~10x lower cost
- **Claude Sonnet 4.6**'s high hallucination rate reflects its verbose style, not factual errors — all extra information was correct but exceeded the ground truth scope. A richer ground truth dataset would reflect its quality more accurately

## Installation

```bash
# Clone the repository
cd HallucinationBenchmark

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env and add your OpenRouter API key
```

## Quick Start

### 1. Create a Sample Dataset

```bash
python cli.py --create-sample --output data/sample.json
```

### 2. Run Benchmark (Mock Mode)

Test the pipeline without API calls:

```bash
python cli.py --dataset data/sample.json --mock-mode --output-dir output/
```

### 3. Run Benchmark with Real Model

```bash
export OPENROUTER_API_KEY=your_key_here
python cli.py --dataset data/sample.json --model openrouter/google/gemini-3-flash-preview
```

## Supported Models

Models are accessed via [OpenRouter](https://openrouter.ai). Pass any OpenRouter model ID with the `openrouter/` prefix.

| Provider | Model | OpenRouter ID |
|----------|-------|--------------|
| Anthropic | Claude Opus 4.6 | `openrouter/anthropic/claude-opus-4-6` |
| Anthropic | Claude Sonnet 4.6 | `openrouter/anthropic/claude-sonnet-4-6` |
| Google | Gemini 3 Pro Preview | `openrouter/google/gemini-3-pro-preview` |
| Google | Gemini 3 Flash Preview | `openrouter/google/gemini-3-flash-preview` |
| OpenAI | GPT-5.4 | `openrouter/openai/gpt-5.4` |
| OpenAI | o3 | `openrouter/openai/o3` |

Any model on OpenRouter can be used — see [openrouter.ai/models](https://openrouter.ai/models) for the full list.

## Usage

### Command Line Interface

```bash
# Run with a specific model
python cli.py --dataset data/benchmark.json --model openrouter/google/gemini-3-flash-preview

# Use a custom judge and output directory
python cli.py --dataset data/benchmark.json --judge-model openrouter/anthropic/claude-opus-4-6 --output-dir results/

# Rule-based only (no LLM judge)
python cli.py --dataset data/benchmark.json --no-llm-judge
```

### Dataset Format

The tool accepts datasets in JSON, JSONL, CSV, or Parquet format:

```json
{
  "name": "my_benchmark",
  "metadata": {"description": "Benchmark dataset"},
  "samples": [
    {
      "id": "sample_1",
      "question": "What is the capital of France?",
      "ground_truth": "The capital of France is Paris.",
      "metadata": {"category": "geography"}
    }
  ]
}
```

CSV format:
```csv
id,question,ground_truth,category
sample_1,What is the capital of France?,The capital of France is Paris.,geography
```

### Output

The tool generates:

- `report.json`: Full evaluation results with per-sample metrics
- `results.csv`: Tabular results for easy analysis
- `report.md`: Human-readable Markdown report with statistics and examples
- `intermediate_results.json`: Checkpoint file during evaluation

## Metrics

### Rule-Based Metrics

- **Faithfulness**: N-gram overlap between response and ground truth
- **Factual Consistency**: Percentage of claims supported by ground truth
- **Hallucination Rate**: Percentage of unsupported claims
- **BLEU Score**: N-gram precision with brevity penalty

### LLM-as-Judge Metrics

- **Semantic Evaluation**: Deep understanding of factual accuracy
- **Hallucination Categories**: Factual error, fabricated citation, unsupported claim
- **Confidence Score**: Judge's confidence in the assessment

## Architecture

```
HallucinationBenchmark/
├── data/               # Dataset loading and validation
│   └── dataset_loader.py
├── model/              # LLM interface
│   └── model_interface.py
├── utils/              # Metrics and LLM judge
│   ├── metrics.py
│   └── llm_judge.py
├── analysis/           # Report generation
│   └── report_generator.py
├── benchmark_runner.py # Main orchestration
├── cli.py             # Command line interface
└── requirements.txt
```

## API Usage

```python
from data.dataset_loader import DatasetLoader
from model.model_interface import OpenRouterModel
from utils.llm_judge import LLMJudge
from benchmark_runner import BenchmarkRunner

# Load dataset
loader = DatasetLoader()
dataset = loader.load("data/benchmark.json")

# Initialize model (model under test)
model = OpenRouterModel(api_key="your_key", model="openrouter/anthropic/claude-sonnet-4-6")

# Initialize judge (can use a different, stronger model)
judge = LLMJudge(api_key="your_key", model="openrouter/anthropic/claude-opus-4-6")

# Run benchmark
runner = BenchmarkRunner(model=model, llm_judge=judge, output_dir="./output")
report = runner.run(dataset)
```

## Requirements

- Python 3.8+
- OpenRouter API key (for LLM evaluation)
- See `requirements.txt` for package dependencies

## License

MIT License

---

> **Built autonomously by [NEO](https://heyneo.so/) — Your Autonomous AI Agent.**
