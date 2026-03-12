# Hallucination Benchmark Tool

[![Powered by](https://img.shields.io/badge/made%20by-NEO-black)](https://marketplace.visualstudio.com/items?itemName=NeoResearchInc.heyneo)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Built autonomously by [NEO](https://heyneo.so/) — Your Autonomous AI Agent.**

A production-ready tool for evaluating Large Language Model (LLM) responses for factual accuracy, fabricated citations, and unsupported claims.

## Features

- **Comprehensive Metrics**: Faithfulness, factual consistency, hallucination rate, BLEU score
- **LLM-as-Judge**: Semantic evaluation using OpenRouter API
- **Rule-Based Checks**: Pattern matching for fabricated citations and claim extraction
- **Modular Architecture**: Easy to swap evaluation metrics and datasets
- **Multiple Input Formats**: JSON, JSONL, CSV, Parquet
- **Detailed Reporting**: JSON, CSV, and Markdown reports with per-sample scores and aggregate statistics
- **✨ Dynamic Benchmarking**: Generate bias-free datasets on-demand using 3-LLM consensus
- **✨ Abstract Parameter Support**: Evaluate traits like 'friendliness', 'technical depth', 'creativity'

## What's New: Bias-Free Dynamic Benchmarking

The tool supports **dynamic dataset generation** with multi-source consensus. Instead of relying on pre-existing datasets that may contain biases, you can generate fresh evaluation data on any topic or abstract parameter:

```bash
# Evaluate friendliness on Gemini 3 Flash
python cli.py --topic "friendliness" --dynamic --model google/gemini-3-flash-preview

# Evaluate technical depth on GPT-5.4
python cli.py --topic "technical depth" --dynamic --model openai/gpt-5.4 --num-entries 10

# Evaluate any topic - factual or abstract
python cli.py --topic "mathematics" --dynamic --model anthropic/claude-sonnet-4-6
```

### How 3-LLM Consensus Works

When using `--dynamic` mode, the tool consults **three distinct LLM providers** to generate ground truth:

1. **OpenAI GPT-5.4** — Latest frontier reasoning & coding model
2. **Anthropic Claude Sonnet 4.6** — Frontier performance across coding, agents, and professional work
3. **Google Gemini 3 Pro Preview** — Enhanced reasoning with 1M-token context

**The Consensus Process:**
1. All three models independently generate responses to the same prompt
2. A judge model (Claude Sonnet 4.6) synthesizes the "ideal" response from the best elements
3. Agreement scoring determines confidence level (high/medium/low)
4. The synthesized response becomes the ground truth for evaluation

**Why This Reduces Bias:**
- **Provider Diversity**: Different training data, architectures, and fine-tuning
- **Independent Generation**: No single model influences the others
- **Synthesis**: The judge extracts the best from all three, not just majority vote
- **Confidence Scoring**: Low agreement flags potentially subjective or ambiguous questions

### Supported Parameter Types

**Factual Topics:**
- Geography, History, Science, Mathematics
- Literature, Arts, Technology, Sports
- Any domain with verifiable facts

**Abstract Parameters/Traits:**
- `friendliness` — Warmth, approachability, positive tone
- `technical depth` — Level of detail, precision, complexity
- `creativity` — Novelty, originality, imaginative thinking
- `empathy` — Understanding, validation, emotional awareness
- `professionalism` — Formality, respect, business etiquette
- `clarity` — Simplicity, understandability, lack of jargon
- `helpfulness` — Actionability, completeness, usefulness

## Live Benchmark Results

Dynamic benchmark on **geography** topic — 3 entries, judge: `anthropic/claude-sonnet-4-6`.

| Model | Hallucination Rate | Faithfulness | Factual Consistency | BLEU | Cost |
|-------|--------------------|--------------|---------------------|------|------|
| **google/gemini-3-flash-preview** | **0.0%** | **100%** | **100%** | 0.000 | $0.0018 |

> Dataset generated via 3-LLM consensus (GPT-5.4 + Claude Sonnet 4.6 + Gemini 3 Pro Preview) with **100% agreement** across all entries.

### Previous Multi-Model Comparison (sample dataset)

| Model | Hallucination Rate | Faithfulness | Factual Consistency | BLEU | Hallucinated | Cost |
|-------|--------------------|--------------|---------------------|------|--------------|------|
| **GPT-5.4** | **13.3%** | **81.9%** | **77.8%** | 0.114 | 1 / 3 | $0.0011 |
| Gemini 3 Flash | 15.0% | 71.8% | 75.0% | **0.362** | 1 / 3 | **$0.0001** |
| Claude Sonnet 4.6 | 46.7% | 12.9% | 22.2% | 0.075 | 3 / 3 | $0.0051 |

### Key Takeaways

- **GPT-5.4** scored best overall — concise answers minimise contextual hallucination flags
- **Gemini 3 Flash** is the best value — near-identical accuracy at ~10x lower cost
- **Claude Sonnet 4.6**'s high hallucination rate reflects its verbose style, not factual errors — all extra information was correct but exceeded the ground truth scope

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

### 1. Dynamic Benchmark (Recommended)

Generate a fresh, bias-free dataset and benchmark any model in one command:

```bash
# Evaluate on 'friendliness' parameter
python cli.py --topic "friendliness" --dynamic --model google/gemini-3-flash-preview

# Evaluate on 'mathematics' topic with 10 samples
python cli.py --topic "mathematics" --dynamic --model openai/gpt-5.4 --num-entries 10

# Save the generated dataset for later use
python cli.py --topic "technical depth" --dynamic --model anthropic/claude-sonnet-4-6 --save-dataset data/tech_depth.json
```

### 2. Standard Benchmark with Existing Dataset

```bash
# Create a sample dataset
python cli.py --create-sample --output data/sample.json

# Run benchmark (Mock Mode - no API calls)
python cli.py --dataset data/sample.json --mock-mode --output-dir output/

# Run benchmark with real model
export OPENROUTER_API_KEY=your_key_here
python cli.py --dataset data/sample.json --model google/gemini-3-flash-preview
```

### 3. Multi-Model Comparison

```bash
# Compare multiple models
python cli.py --dataset data/benchmark.json --model openai/gpt-5.4 anthropic/claude-sonnet-4-6 google/gemini-3-flash-preview

# Dynamic mode with multiple targets
python cli.py --topic "creativity" --dynamic --model openai/gpt-5.4 anthropic/claude-sonnet-4-6
```

## Supported Models

Models are accessed via [OpenRouter](https://openrouter.ai). Pass any OpenRouter model ID.

| Provider | Model | OpenRouter ID |
|----------|-------|--------------|
| OpenAI | GPT-5.4 | `openai/gpt-5.4` |
| OpenAI | GPT-5.3 | `openai/gpt-5.3` |
| OpenAI | GPT-5.2 | `openai/gpt-5.2` |
| Anthropic | Claude Sonnet 4.6 | `anthropic/claude-sonnet-4-6` |
| Anthropic | Claude Opus 4.6 | `anthropic/claude-opus-4-6` |
| Google | Gemini 3 Pro Preview | `google/gemini-3-pro-preview` |
| Google | Gemini 3 Flash Preview | `google/gemini-3-flash-preview` |
| Google | Gemini 2.5 Pro | `google/gemini-2.5-pro` |

Any model on OpenRouter can be used — see [openrouter.ai/models](https://openrouter.ai/models) for the full list.

## Usage

### Command Line Interface

#### Dynamic Mode (Generate + Benchmark)

```bash
# Basic dynamic benchmark
python cli.py --topic "friendliness" --dynamic --model google/gemini-3-flash-preview

# With custom options
python cli.py --topic "mathematics" --dynamic --model openai/gpt-5.4 \
    --num-entries 10 \
    --judge-model anthropic/claude-sonnet-4-6 \
    --output-dir results/ \
    --save-dataset data/generated.json
```

#### Standard Mode (Existing Dataset)

```bash
# Run with a specific model
python cli.py --dataset data/benchmark.json --model google/gemini-3-flash-preview

# Use a custom judge and output directory
python cli.py --dataset data/benchmark.json --judge-model anthropic/claude-sonnet-4-6 --output-dir results/

# Rule-based only (no LLM judge)
python cli.py --dataset data/benchmark.json --no-llm-judge
```

### CLI Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--topic` | Topic or parameter for generation | — |
| `--dynamic` | Enable dynamic generation + benchmark mode | False |
| `--num-entries` | Number of samples to generate | 5 |
| `--model` | Target model(s) to evaluate | `google/gemini-3-flash-preview` |
| `--dataset` | Path to existing dataset | — |
| `--judge-model` | Model for LLM-as-Judge | `anthropic/claude-sonnet-4-6` |
| `--output-dir` | Output directory for results | `./output` |
| `--save-dataset` | Save generated dataset to path | — |
| `--mock-mode` | Run without API calls | False |
| `--no-llm-judge` | Disable LLM-as-Judge | False |

### Dataset Format

The tool accepts datasets in JSON, JSONL, CSV, or Parquet format:

```json
{
  "name": "my_benchmark",
  "metadata": {
    "description": "Benchmark dataset",
    "parameter_type": "factual",
    "source_models": ["openai/gpt-5.4", "anthropic/claude-sonnet-4-6", "google/gemini-3-pro-preview"]
  },
  "samples": [
    {
      "id": "sample_1",
      "question": "What is the capital of France?",
      "ground_truth": "The capital of France is Paris.",
      "metadata": {
        "category": "geography",
        "consensus_score": 0.95,
        "agreement_level": "high"
      }
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

- `report.json` — Full evaluation results with per-sample metrics
- `results.csv` — Tabular results for easy analysis
- `report.md` — Human-readable Markdown report with statistics and examples
- `intermediate_results.json` — Checkpoint file during evaluation

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
├── data/                         # Dataset loading and generation
│   ├── dataset_loader.py          # Load/validate datasets
│   └── multi_source_generator.py  # 3-LLM consensus generation
├── model/                        # LLM interface
│   └── model_interface.py         # OpenRouter API wrapper
├── utils/                        # Metrics and LLM judge
│   ├── metrics.py                 # Hallucination metrics
│   └── llm_judge.py               # LLM-as-Judge evaluation
├── analysis/                     # Report generation
│   └── report_generator.py        # Report generators
├── benchmark_runner.py           # Main orchestration
├── cli.py                        # Command line interface
└── requirements.txt
```

## API Usage

```python
import os
from data.dataset_loader import DatasetLoader
from data.multi_source_generator import MultiSourceGenerator
from model.model_interface import OpenRouterModel
from utils.llm_judge import LLMJudge
from benchmark_runner import BenchmarkRunner

api_key = os.getenv("OPENROUTER_API_KEY")

# Dynamic dataset generation (3-LLM consensus)
generator = MultiSourceGenerator(api_key=api_key)
dataset = generator.generate_dataset(
    topic="friendliness",
    num_entries=5,
    output_path="data/friendliness.json"
)

# Load dataset
loader = DatasetLoader()
benchmark_dataset = loader.load("data/friendliness.json")

# Initialize model under test
model = OpenRouterModel(api_key=api_key, model="google/gemini-3-flash-preview")

# Initialize judge
judge = LLMJudge(api_key=api_key, model="anthropic/claude-sonnet-4-6")

# Run benchmark
runner = BenchmarkRunner(model=model, llm_judge=judge, output_dir="./output")
report = runner.run(benchmark_dataset)
```

## Requirements

- Python 3.8+
- OpenRouter API key (for LLM evaluation)
- See `requirements.txt` for package dependencies

## License

MIT License

> **Built autonomously by [NEO](https://heyneo.so/) — Your Autonomous AI Agent.**
