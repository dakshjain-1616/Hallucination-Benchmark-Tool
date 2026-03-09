"""
Report generator for hallucination benchmark results.
Produces structured output with per-sample scores, aggregate statistics, and examples.
"""

import json
import os
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict, field
from datetime import datetime
import pandas as pd
import numpy as np


@dataclass
class SampleResult:
    """Result for a single sample."""
    sample_id: str
    question: str
    response: str
    ground_truth: str
    faithfulness: float
    factual_consistency: float
    hallucination_rate: float
    bleu_score: float
    claim_count: int
    citation_count: int
    llm_judge_score: Optional[float] = None
    llm_judge_reasoning: Optional[str] = None
    hallucination_detected: bool = False
    hallucination_category: Optional[str] = None


@dataclass
class AggregateStats:
    """Aggregate statistics across all samples."""
    total_samples: int
    avg_faithfulness: float
    avg_factual_consistency: float
    avg_hallucination_rate: float
    avg_bleu_score: float
    hallucination_detected_count: int
    hallucination_percentage: float
    avg_claims_per_response: float
    avg_citations_per_response: float
    category_distribution: Dict[str, int] = field(default_factory=dict)


@dataclass
class BenchmarkReport:
    """Complete benchmark report."""
    report_id: str
    timestamp: str
    model_name: str
    dataset_name: str
    sample_results: List[SampleResult]
    aggregate_stats: AggregateStats
    config: Dict[str, Any]
    examples: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary."""
        return asdict(self)
    
    def to_json(self, filepath: str) -> str:
        """Save report to JSON file."""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        return filepath


class ReportGenerator:
    """Generate structured benchmark reports."""
    
    def __init__(self, output_dir: str = "./analysis/reports"):
        """
        Initialize report generator.
        
        Args:
            output_dir: Directory for output reports
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def generate_report(
        self,
        results: List[Dict[str, Any]],
        model_name: str,
        dataset_name: str,
        config: Optional[Dict[str, Any]] = None
    ) -> BenchmarkReport:
        """
        Generate a complete benchmark report.
        
        Args:
            results: List of evaluation results per sample
            model_name: Name of the evaluated model
            dataset_name: Name of the dataset
            config: Configuration used for evaluation
            
        Returns:
            BenchmarkReport object
        """
        # Create sample results
        sample_results = []
        for result in results:
            sample_result = SampleResult(
                sample_id=result.get('sample_id', 'unknown'),
                question=result.get('question', ''),
                response=result.get('response', ''),
                ground_truth=result.get('ground_truth', ''),
                faithfulness=result.get('faithfulness', 0.0),
                factual_consistency=result.get('factual_consistency', 0.0),
                hallucination_rate=result.get('hallucination_rate', 0.0),
                bleu_score=result.get('bleu_score', 0.0),
                claim_count=result.get('claim_count', 0),
                citation_count=result.get('citation_count', 0),
                llm_judge_score=result.get('llm_judge_score'),
                llm_judge_reasoning=result.get('llm_judge_reasoning'),
                hallucination_detected=result.get('hallucination_detected', False),
                hallucination_category=result.get('hallucination_category')
            )
            sample_results.append(sample_result)
        
        # Calculate aggregate statistics
        aggregate_stats = self._calculate_aggregate_stats(sample_results)
        
        # Extract examples
        examples = self._extract_examples(sample_results)
        
        # Create report
        report = BenchmarkReport(
            report_id=f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            timestamp=datetime.now().isoformat(),
            model_name=model_name,
            dataset_name=dataset_name,
            sample_results=sample_results,
            aggregate_stats=aggregate_stats,
            config=config or {},
            examples=examples
        )
        
        return report
    
    def _calculate_aggregate_stats(self, sample_results: List[SampleResult]) -> AggregateStats:
        """Calculate aggregate statistics from sample results."""
        if not sample_results:
            return AggregateStats(
                total_samples=0,
                avg_faithfulness=0.0,
                avg_factual_consistency=0.0,
                avg_hallucination_rate=0.0,
                avg_bleu_score=0.0,
                hallucination_detected_count=0,
                hallucination_percentage=0.0,
                avg_claims_per_response=0.0,
                avg_citations_per_response=0.0
            )
        
        total = len(sample_results)
        hallucinated = sum(1 for s in sample_results if s.hallucination_detected)
        
        # Category distribution
        categories = {}
        for s in sample_results:
            if s.hallucination_category:
                categories[s.hallucination_category] = categories.get(s.hallucination_category, 0) + 1
        
        return AggregateStats(
            total_samples=total,
            avg_faithfulness=round(sum(s.faithfulness for s in sample_results) / total, 4),
            avg_factual_consistency=round(sum(s.factual_consistency for s in sample_results) / total, 4),
            avg_hallucination_rate=round(sum(s.hallucination_rate for s in sample_results) / total, 4),
            avg_bleu_score=round(sum(s.bleu_score for s in sample_results) / total, 4),
            hallucination_detected_count=hallucinated,
            hallucination_percentage=round(hallucinated / total * 100, 2),
            avg_claims_per_response=round(sum(s.claim_count for s in sample_results) / total, 2),
            avg_citations_per_response=round(sum(s.citation_count for s in sample_results) / total, 2),
            category_distribution=categories
        )
    
    def _extract_examples(self, sample_results: List[SampleResult]) -> Dict[str, List[Dict[str, Any]]]:
        """Extract example cases for different categories."""
        examples = {
            'hallucinated': [],
            'factual_error': [],
            'fabricated_citation': [],
            'unsupported_claim': [],
            'correct': []
        }
        
        for sample in sample_results:
            example = {
                'sample_id': sample.sample_id,
                'question': sample.question,
                'response': sample.response,
                'ground_truth': sample.ground_truth,
                'score': sample.llm_judge_score,
                'reasoning': sample.llm_judge_reasoning
            }
            
            if sample.hallucination_detected:
                examples['hallucinated'].append(example)
                if sample.hallucination_category:
                    cat = sample.hallucination_category
                    if cat in examples:
                        if example not in examples[cat]:
                            examples[cat].append(example)
            else:
                examples['correct'].append(example)
        
        # Limit examples to top 3 per category
        for key in examples:
            examples[key] = examples[key][:3]
        
        return examples
    
    def export_report(
        self,
        report: BenchmarkReport,
        formats: List[str] = ['json', 'csv', 'html'],
        output_dir: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Export report in multiple formats.
        
        Args:
            report: BenchmarkReport to export
            formats: List of formats to export ('json', 'csv', 'html', 'markdown')
            output_dir: Output directory (defaults to self.output_dir)
            
        Returns:
            Dictionary mapping format to file path
        """
        output_dir = output_dir or self.output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        base_name = f"benchmark_report_{report.report_id}"
        exported = {}
        
        if 'json' in formats:
            json_path = os.path.join(output_dir, f"{base_name}.json")
            report.to_json(json_path)
            exported['json'] = json_path
        
        if 'csv' in formats:
            csv_path = os.path.join(output_dir, f"{base_name}.csv")
            self._export_csv(report, csv_path)
            exported['csv'] = csv_path
        
        if 'html' in formats:
            html_path = os.path.join(output_dir, f"{base_name}.html")
            self._export_html(report, html_path)
            exported['html'] = html_path
        
        if 'markdown' in formats:
            md_path = os.path.join(output_dir, f"{base_name}.md")
            self._export_markdown(report, md_path)
            exported['markdown'] = md_path
        
        return exported
    
    def _export_csv(self, report: BenchmarkReport, filepath: str):
        """Export sample results to CSV."""
        data = []
        for sample in report.sample_results:
            data.append({
                'sample_id': sample.sample_id,
                'question': sample.question,
                'response': sample.response,
                'ground_truth': sample.ground_truth,
                'faithfulness': sample.faithfulness,
                'factual_consistency': sample.factual_consistency,
                'hallucination_rate': sample.hallucination_rate,
                'bleu_score': sample.bleu_score,
                'claim_count': sample.claim_count,
                'citation_count': sample.citation_count,
                'llm_judge_score': sample.llm_judge_score,
                'hallucination_detected': sample.hallucination_detected,
                'hallucination_category': sample.hallucination_category
            })
        
        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False)
    
    def _export_html(self, report: BenchmarkReport, filepath: str):
        """Export report to HTML."""
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Hallucination Benchmark Report - {report.report_id}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        h1 {{ color: #333; border-bottom: 2px solid #4CAF50; padding-bottom: 10px; }}
        h2 {{ color: #555; margin-top: 30px; }}
        .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }}
        .stat-card {{ background: #f9f9f9; padding: 15px; border-radius: 6px; border-left: 4px solid #4CAF50; }}
        .stat-label {{ font-size: 12px; color: #666; text-transform: uppercase; }}
        .stat-value {{ font-size: 24px; font-weight: bold; color: #333; }}
        .sample {{ background: #f9f9f9; padding: 15px; margin: 10px 0; border-radius: 6px; border-left: 4px solid #2196F3; }}
        .sample.hallucinated {{ border-left-color: #f44336; background: #ffebee; }}
        .sample.correct {{ border-left-color: #4CAF50; background: #e8f5e9; }}
        .metric-row {{ display: flex; gap: 20px; margin: 10px 0; flex-wrap: wrap; }}
        .metric {{ background: white; padding: 8px 12px; border-radius: 4px; font-size: 14px; }}
        .metric-label {{ color: #666; }}
        .metric-value {{ font-weight: bold; color: #333; }}
        pre {{ background: #f4f4f4; padding: 10px; border-radius: 4px; overflow-x: auto; white-space: pre-wrap; }}
        .timestamp {{ color: #999; font-size: 14px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Hallucination Benchmark Report</h1>
        <p class="timestamp">Generated: {report.timestamp}</p>
        <p><strong>Report ID:</strong> {report.report_id}</p>
        <p><strong>Model:</strong> {report.model_name}</p>
        <p><strong>Dataset:</strong> {report.dataset_name}</p>
        
        <h2>Aggregate Statistics</h2>
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-label">Total Samples</div>
                <div class="stat-value">{report.aggregate_stats.total_samples}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Avg Faithfulness</div>
                <div class="stat-value">{report.aggregate_stats.avg_faithfulness:.2%}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Avg Factual Consistency</div>
                <div class="stat-value">{report.aggregate_stats.avg_factual_consistency:.2%}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Hallucination Rate</div>
                <div class="stat-value">{report.aggregate_stats.avg_hallucination_rate:.2%}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Hallucinated Samples</div>
                <div class="stat-value">{report.aggregate_stats.hallucination_detected_count} ({report.aggregate_stats.hallucination_percentage:.1f}%)</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Avg BLEU Score</div>
                <div class="stat-value">{report.aggregate_stats.avg_bleu_score:.4f}</div>
            </div>
        </div>
        
        <h2>Category Distribution</h2>
        <div class="stats-grid">
"""
        
        # Add category distribution
        for category, count in report.aggregate_stats.category_distribution.items():
            html += f"""
            <div class="stat-card">
                <div class="stat-label">{category.replace('_', ' ').title()}</div>
                <div class="stat-value">{count}</div>
            </div>
"""
        
        html += """
        </div>
        
        <h2>Sample Results</h2>
"""
        
        # Add sample results
        for sample in report.sample_results:
            css_class = "hallucinated" if sample.hallucination_detected else "correct"
            html += f"""
        <div class="sample {css_class}">
            <h3>Sample {sample.sample_id}</h3>
            <p><strong>Question:</strong> {sample.question}</p>
            <div class="metric-row">
                <div class="metric"><span class="metric-label">Faithfulness:</span> <span class="metric-value">{sample.faithfulness:.2%}</span></div>
                <div class="metric"><span class="metric-label">Factual Consistency:</span> <span class="metric-value">{sample.factual_consistency:.2%}</span></div>
                <div class="metric"><span class="metric-label">Hallucination Rate:</span> <span class="metric-value">{sample.hallucination_rate:.2%}</span></div>
                <div class="metric"><span class="metric-label">BLEU:</span> <span class="metric-value">{sample.bleu_score:.4f}</span></div>
            </div>
            <p><strong>Response:</strong></p>
            <pre>{sample.response}</pre>
            <p><strong>Ground Truth:</strong></p>
            <pre>{sample.ground_truth}</pre>
        </div>
"""
        
        html += """
    </div>
</body>
</html>
"""
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html)
    
    def _export_markdown(self, report: BenchmarkReport, filepath: str):
        """Export report to Markdown file."""
        md = self._generate_markdown(report)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(md)
    
    def _generate_markdown(self, report: BenchmarkReport) -> str:
        """Generate Markdown report."""
        md = f"""# Hallucination Benchmark Report

**Report ID:** {report.report_id}  
**Generated:** {report.timestamp}  
**Model:** {report.model_name}  
**Dataset:** {report.dataset_name}

## Aggregate Statistics

| Metric | Value |
|--------|-------|
| Total Samples | {report.aggregate_stats.total_samples} |
| Avg Faithfulness | {report.aggregate_stats.avg_faithfulness:.2%} |
| Avg Factual Consistency | {report.aggregate_stats.avg_factual_consistency:.2%} |
| Avg Hallucination Rate | {report.aggregate_stats.avg_hallucination_rate:.2%} |
| Avg BLEU Score | {report.aggregate_stats.avg_bleu_score:.4f} |
| Hallucinated Samples | {report.aggregate_stats.hallucination_detected_count} ({report.aggregate_stats.hallucination_percentage:.1f}%) |
| Avg Claims per Response | {report.aggregate_stats.avg_claims_per_response:.2f} |
| Avg Citations per Response | {report.aggregate_stats.avg_citations_per_response:.2f} |

## Category Distribution

"""
        for category, count in report.aggregate_stats.category_distribution.items():
            md += f"- **{category.replace('_', ' ').title()}:** {count}\n"
        
        md += "\n## Sample Results\n\n"
        
        for sample in report.sample_results:
            status = "❌ HALLUCINATED" if sample.hallucination_detected else "✅ CORRECT"
            md += f"""### Sample {sample.sample_id} {status}

**Question:** {sample.question}

**Metrics:**
- Faithfulness: {sample.faithfulness:.2%}
- Factual Consistency: {sample.factual_consistency:.2%}
- Hallucination Rate: {sample.hallucination_rate:.2%}
- BLEU Score: {sample.bleu_score:.4f}

**Response:**
```
{sample.response}
```

**Ground Truth:**
```
{sample.ground_truth}
```

"""
        return md
    
    def generate_comparison_report(self, model_reports: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate a comparison report for multiple models.
        
        Args:
            model_reports: List of dicts containing 'model_name', 'report', and 'output_dir'
            
        Returns:
            Dictionary containing comparison data
        """
        comparison_data = {
            'timestamp': datetime.now().isoformat(),
            'model_count': len(model_reports),
            'model_rankings': [],
            'side_by_side': []
        }
        
        # Extract key metrics for each model
        for model_data in model_reports:
            model_name = model_data['model_name']
            report = model_data['report']
            agg = report.get('aggregate_stats', {})
            config = report.get('config', {})
            usage = config.get('usage_stats', {})
            
            model_summary = {
                'model_name': model_name,
                'avg_faithfulness': agg.get('avg_faithfulness', 0),
                'avg_factual_consistency': agg.get('avg_factual_consistency', 0),
                'avg_hallucination_rate': agg.get('avg_hallucination_rate', 0),
                'avg_bleu_score': agg.get('avg_bleu_score', 0),
                'hallucination_percentage': agg.get('hallucination_percentage', 0),
                'total_samples': agg.get('total_samples', 0),
                'prompt_tokens': usage.get('prompt_tokens', 0),
                'completion_tokens': usage.get('completion_tokens', 0),
                'total_tokens': usage.get('total_tokens', 0),
                'cache_read_tokens': usage.get('cache_read_tokens', 0),
                'cache_write_tokens': usage.get('cache_write_tokens', 0),
                'total_cost': usage.get('total_cost', 0),
                'request_count': usage.get('request_count', 0)
            }
            comparison_data['model_rankings'].append(model_summary)
        
        # Sort by hallucination rate (ascending - lower is better)
        comparison_data['model_rankings'].sort(key=lambda x: x['avg_hallucination_rate'])
        
        # Generate comparison files
        self._export_comparison_json(comparison_data)
        self._export_comparison_markdown(comparison_data)
        
        return comparison_data
    
    def _export_comparison_json(self, comparison_data: Dict[str, Any]):
        """Export comparison data to JSON."""
        filepath = os.path.join(self.output_dir, 'comparison_report.json')
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(comparison_data, f, indent=2, default=str)
    
    def _export_comparison_markdown(self, comparison_data: Dict[str, Any]):
        """Export comparison data to Markdown."""
        md = f"""# Model Comparison Report

**Generated:** {comparison_data['timestamp']}  
**Models Compared:** {comparison_data['model_count']}

## Model Rankings (by Hallucination Rate)

| Rank | Model | Hallucination Rate | Faithfulness | Factual Consistency | BLEU Score | Total Cost |
|------|-------|-------------------|--------------|---------------------|------------|------------|
"""
        
        for i, model in enumerate(comparison_data['model_rankings'], 1):
            md += f"| {i} | {model['model_name']} | {model['avg_hallucination_rate']:.2%} | {model['avg_faithfulness']:.2%} | {model['avg_factual_consistency']:.2%} | {model['avg_bleu_score']:.4f} | ${model['total_cost']:.4f} |\n"
        
        md += f"""
## Usage by Model

| Model | Input Tokens | Output Tokens | Cache Read | Cache Write | Total Tokens | Cost |
|-------|-------------|---------------|------------|-------------|--------------|------|
"""
        
        for model in comparison_data['model_rankings']:
            md += f"| {model['model_name']} | {model['prompt_tokens']:,} | {model['completion_tokens']:,} | {model['cache_read_tokens']:,} | {model['cache_write_tokens']:,} | {model['total_tokens']:,} | ${model['total_cost']:.4f} |\n"
        
        md += """
## Summary

Models are ranked by hallucination rate (lower is better). Cost includes both input and output token pricing, plus cache read/write costs where applicable.

"""
        
        filepath = os.path.join(self.output_dir, 'comparison_report.md')
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(md)