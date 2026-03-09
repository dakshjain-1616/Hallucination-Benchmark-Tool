#!/usr/bin/env python3
"""
Command Line Interface for Hallucination Benchmark Tool.

Usage:
    python cli.py --dataset path/to/dataset.json --model openrouter/google/gemini-3-flash-preview
    python cli.py --dataset path/to/dataset.json --mock-mode
    python cli.py --create-sample --output path/to/sample.json
"""

import argparse
import os
import sys
import json
from typing import Optional, List, Dict, Any

from data.dataset_loader import DatasetLoader
from model.model_interface import OpenRouterModel, MockModel
from utils.llm_judge import LLMJudge
from benchmark_runner import BenchmarkRunner


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        description="Hallucination Benchmark Tool - Evaluate LLM responses for factual accuracy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run benchmark with OpenRouter model
  python cli.py --dataset data/benchmark.json --model google/gemini-3-flash-preview

  # Run in mock mode (no API calls)
  python cli.py --dataset data/benchmark.json --mock-mode

  # Create sample dataset
  python cli.py --create-sample --output data/sample.json

  # Specify output directory
  python cli.py --dataset data/benchmark.json --output-dir results/
        """
    )
    
    # Required arguments
    parser.add_argument(
        '--dataset',
        type=str,
        help='Path to benchmark dataset (JSON, JSONL, CSV, or Parquet)'
    )
    
    # Model configuration
    model_group = parser.add_argument_group('Model Configuration')
    model_group.add_argument(
        '--model',
        type=str,
        nargs='+',
        default=['google/gemini-3-flash-preview'],
        help='Model identifier(s) for OpenRouter. Can specify multiple for comparison (default: google/gemini-3-flash-preview)'
    )
    model_group.add_argument(
        '--mock-mode',
        action='store_true',
        help='Run in mock mode without making API calls'
    )
    model_group.add_argument(
        '--temperature',
        type=float,
        default=0.7,
        help='Temperature for generation (default: 0.7)'
    )
    model_group.add_argument(
        '--max-tokens',
        type=int,
        default=1024,
        help='Maximum tokens for generation (default: 1024)'
    )
    
    # Evaluation configuration
    eval_group = parser.add_argument_group('Evaluation Configuration')
    eval_group.add_argument(
        '--no-llm-judge',
        action='store_true',
        help='Disable LLM-as-Judge evaluation'
    )
    eval_group.add_argument(
        '--judge-model',
        type=str,
        default='anthropic/claude-opus-4-6',
        help='Model for LLM-as-Judge (default: anthropic/claude-opus-4-6)'
    )
    
    # Output configuration
    output_group = parser.add_argument_group('Output Configuration')
    output_group.add_argument(
        '--output-dir',
        type=str,
        default='./output',
        help='Output directory for results (default: ./output)'
    )
    output_group.add_argument(
        '--no-intermediate',
        action='store_true',
        help='Disable intermediate result saving'
    )
    
    # Utility commands
    util_group = parser.add_argument_group('Utility Commands')
    util_group.add_argument(
        '--create-sample',
        action='store_true',
        help='Create a sample dataset file'
    )
    util_group.add_argument(
        '--output',
        type=str,
        help='Output path for sample dataset (used with --create-sample)'
    )
    util_group.add_argument(
        '--format',
        type=str,
        default='json',
        choices=['json', 'jsonl', 'csv'],
        help='Format for sample dataset (default: json)'
    )
    
    # API configuration
    api_group = parser.add_argument_group('API Configuration')
    api_group.add_argument(
        '--api-key',
        type=str,
        help='OpenRouter API key (or set OPENROUTER_API_KEY env var)'
    )
    
    return parser


def validate_args(args: argparse.Namespace) -> bool:
    """Validate command line arguments."""
    if args.create_sample:
        if not args.output:
            print("Error: --output required when using --create-sample")
            return False
        return True
    
    # Auto-create data directory if it doesn't exist
    data_dir = "./data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)
        print(f"Created data directory: {data_dir}")
    
    # Handle dataset fallback
    if not args.dataset:
        # Try default paths in order of preference
        default_paths = ['data/benchmark.json', 'data/sample.json']
        for path in default_paths:
            if os.path.exists(path):
                args.dataset = path
                print(f"Using default dataset: {path}")
                break
        
        if not args.dataset:
            # Create sample.json if neither exists
            args.dataset = 'data/sample.json'
            print(f"No dataset specified. Creating and using sample dataset: {args.dataset}")
            loader = DatasetLoader()
            loader.create_sample_dataset(args.dataset, 'json')
    else:
        # Check if specified dataset exists, fallback to sample if not
        if not os.path.exists(args.dataset):
            if args.dataset == 'data/benchmark.json':
                fallback_path = 'data/sample.json'
                if os.path.exists(fallback_path):
                    print(f"Dataset not found: {args.dataset}")
                    print(f"Falling back to: {fallback_path}")
                    args.dataset = fallback_path
                else:
                    print(f"Dataset not found: {args.dataset}")
                    print(f"Creating sample dataset: {fallback_path}")
                    loader = DatasetLoader()
                    loader.create_sample_dataset(fallback_path, 'json')
                    args.dataset = fallback_path
            else:
                print(f"Error: Dataset file not found: {args.dataset}")
                return False
    
    if not args.mock_mode:
        api_key = args.api_key or os.getenv('OPENROUTER_API_KEY')
        if not api_key:
            print("Error: OpenRouter API key required. Use --api-key or set OPENROUTER_API_KEY env var")
            return False
    
    return True


def create_sample_dataset(args: argparse.Namespace):
    """Create a sample dataset."""
    loader = DatasetLoader()
    filepath = loader.create_sample_dataset(args.output, args.format)
    print(f"Sample dataset created: {filepath}")
    
    # Display sample content
    with open(filepath, 'r') as f:
        if args.format == 'json':
            data = json.load(f)
            print(f"\nDataset contains {len(data.get('samples', []))} samples")
        else:
            content = f.read()
            lines = content.strip().split('\n')
            print(f"\nDataset contains {len(lines)} samples")


def run_benchmark(args: argparse.Namespace):
    """Run the benchmark."""
    print("=" * 60)
    print("Hallucination Benchmark Tool")
    print("=" * 60)
    
    # Load dataset
    print(f"\nLoading dataset: {args.dataset}")
    loader = DatasetLoader()
    dataset = loader.load(args.dataset)
    print(f"Loaded {len(dataset)} samples")
    
    # Handle multiple models for comparison
    models = args.model if isinstance(args.model, list) else [args.model]
    
    # Store results for each model
    all_model_reports = []
    
    for model_idx, model_name in enumerate(models):
        print(f"\n{'=' * 60}")
        print(f"Evaluating Model {model_idx + 1}/{len(models)}: {model_name}")
        print(f"{'=' * 60}")
        
        # Initialize model
        if args.mock_mode:
            print("\nRunning in MOCK MODE (no API calls)")
            model = MockModel([
                "Paris is the capital of France.",
                "William Shakespeare wrote Romeo and Juliet in 1597.",
                "The speed of light is 300,000 km/s."
            ])
        else:
            api_key = args.api_key or os.getenv('OPENROUTER_API_KEY')
            print(f"\nInitializing model: {model_name}")
            model = OpenRouterModel(
                api_key=api_key,
                model=model_name
            )
        
        # Initialize LLM judge
        llm_judge = None
        if not args.no_llm_judge and not args.mock_mode:
            print(f"Initializing LLM judge: {args.judge_model}")
            api_key = args.api_key or os.getenv('OPENROUTER_API_KEY')
            llm_judge = LLMJudge(
                api_key=api_key,
                model=args.judge_model
            )
        elif args.mock_mode:
            print("LLM judge disabled in mock mode")
        else:
            print("LLM judge disabled")
        
        # Initialize runner
        model_output_dir = os.path.join(args.output_dir, model_name.replace('/', '_')) if len(models) > 1 else args.output_dir
        print(f"\nOutput directory: {model_output_dir}")
        runner = BenchmarkRunner(
            model=model,
            llm_judge=llm_judge,
            output_dir=model_output_dir,
            use_llm_judge=not args.no_llm_judge and not args.mock_mode
        )
        
        # Run benchmark
        print("\n" + "=" * 60)
        print("Running Benchmark")
        print("=" * 60)
        
        report = runner.run(
            dataset=dataset,
            save_intermediate=not args.no_intermediate
        )
        
        # Store report for comparison
        all_model_reports.append({
            'model_name': model_name,
            'report': report,
            'output_dir': model_output_dir
        })
        
        # Display summary for this model
        print("\n" + "=" * 60)
        print(f"Benchmark Complete for {model_name}")
        print("=" * 60)
        config = report.get('config', {})
        agg = report.get('aggregate_stats', {})
        print(f"\nTotal samples: {config.get('total_samples', agg.get('total_samples', 0))}")
        print(f"Successful: {config.get('successful_samples', 0)}")
        print(f"Failed: {config.get('failed_samples', 0)}")

        print(f"\nAggregate Metrics:")
        print(f"  Average Faithfulness: {agg.get('avg_faithfulness', 0):.4f}")
        print(f"  Average Factual Consistency: {agg.get('avg_factual_consistency', 0):.4f}")
        print(f"  Average Hallucination Rate: {agg.get('avg_hallucination_rate', 0):.4f}")
        print(f"  Average BLEU Score: {agg.get('avg_bleu_score', 0):.4f}")
        print(f"  Hallucinated Samples: {agg.get('hallucination_detected_count', 0)} ({agg.get('hallucination_percentage', 0):.1f}%)")
        
        print(f"\nReports saved to: {model_output_dir}")
        print(f"  - Full report: {os.path.join(model_output_dir, 'report.json')}")
        print(f"  - CSV results: {os.path.join(model_output_dir, 'results.csv')}")
        print(f"  - Markdown report: {os.path.join(model_output_dir, 'report.md')}")
    
    # Generate comparison report if multiple models
    if len(all_model_reports) > 1:
        print("\n" + "=" * 60)
        print("Generating Model Comparison Report")
        print("=" * 60)
        generate_comparison_report(all_model_reports, args.output_dir)


def generate_comparison_report(all_model_reports: List[Dict], output_dir: str):
    """Generate a comparison report for multiple models."""
    from analysis.report_generator import ReportGenerator
    
    report_gen = ReportGenerator(output_dir=output_dir)
    comparison_report = report_gen.generate_comparison_report(all_model_reports)
    
    print(f"\nComparison report saved to: {output_dir}")
    print(f"  - Comparison JSON: {os.path.join(output_dir, 'comparison_report.json')}")
    print(f"  - Comparison Markdown: {os.path.join(output_dir, 'comparison_report.md')}")
    
    # Print comparison summary
    print("\n" + "=" * 60)
    print("Model Comparison Summary")
    print("=" * 60)
    print(f"{'Model':<40} {'Hallucination Rate':<18} {'Faithfulness':<14} {'Cost':<10}")
    print("-" * 60)
    
    for model_data in comparison_report.get('model_rankings', []):
        model_name = model_data['model_name'][:38]
        hall_rate = model_data.get('avg_hallucination_rate', 0)
        faith = model_data.get('avg_faithfulness', 0)
        cost = model_data.get('total_cost', 0)
        print(f"{model_name:<40} {hall_rate:<18.2%} {faith:<14.2%} ${cost:<10.4f}")


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)
    
    if not validate_args(args):
        sys.exit(1)
    
    try:
        if args.create_sample:
            create_sample_dataset(args)
        else:
            run_benchmark(args)
    except KeyboardInterrupt:
        print("\n\nBenchmark interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
