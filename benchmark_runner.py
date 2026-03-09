"""
Benchmark runner module for hallucination evaluation.
Orchestrates the end-to-end evaluation pipeline.
"""

import os
import json
import time
import logging
from typing import Dict, List, Any, Optional
from dataclasses import asdict
from datetime import datetime

from data.dataset_loader import DatasetLoader, BenchmarkDataset, BenchmarkSample
from model.model_interface import ModelInterface, OpenRouterModel, MockModel
from utils.metrics import HallucinationMetrics
from utils.llm_judge import LLMJudge, JudgeResult
from analysis.report_generator import ReportGenerator


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """Main benchmark runner for hallucination evaluation."""
    
    def __init__(
        self,
        model: ModelInterface,
        metrics: Optional[HallucinationMetrics] = None,
        llm_judge: Optional[LLMJudge] = None,
        output_dir: str = "./output",
        use_llm_judge: bool = True
    ):
        """
        Initialize the benchmark runner.
        
        Args:
            model: Model interface for generating responses
            metrics: Metrics calculator (creates default if None)
            llm_judge: LLM judge for semantic evaluation
            output_dir: Directory for output files
            use_llm_judge: Whether to use LLM-as-judge evaluation
        """
        self.model = model
        self.metrics = metrics or HallucinationMetrics()
        self.llm_judge = llm_judge
        self.output_dir = output_dir
        self.use_llm_judge = use_llm_judge
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Results storage
        self.results = []
        self.start_time = None
        self.end_time = None
    
    def run(
        self,
        dataset: BenchmarkDataset,
        save_intermediate: bool = True,
    ) -> Dict[str, Any]:
        """
        Run the benchmark on a dataset.
        
        Args:
            dataset: Benchmark dataset to evaluate
            save_intermediate: Whether to save results after each sample
            
        Returns:
            Dictionary containing evaluation results
        """
        logger.info(f"Starting benchmark on dataset: {dataset.name}")
        logger.info(f"Number of samples: {len(dataset)}")
        
        self.start_time = datetime.now()
        self.results = []
        
        for i, sample in enumerate(dataset.samples):
            logger.info(f"Processing sample {i+1}/{len(dataset)}: {sample.sample_id}")
            
            try:
                result = self._evaluate_sample(sample)
                self.results.append(result)
                
                if save_intermediate:
                    self._save_intermediate_results()
                    
            except Exception as e:
                logger.error(f"Error evaluating sample {sample.sample_id}: {e}")
                # Add error result
                self.results.append({
                    'sample_id': sample.sample_id,
                    'error': str(e),
                    'status': 'failed'
                })
        
        self.end_time = datetime.now()

        # Generate final report
        report = self._generate_report(dataset.name)
        logger.info("Benchmark completed successfully")

        return report
    
    def _evaluate_sample(self, sample: BenchmarkSample) -> Dict[str, Any]:
        """Evaluate a single sample."""
        # Generate response if not provided
        if sample.response is None:
            logger.debug(f"Generating response for sample {sample.sample_id}")
            prompt = f"Question: {sample.question}\nAnswer:"
            response = self.model.generate(prompt)
        else:
            response = sample.response
        
        # Calculate rule-based metrics
        logger.debug(f"Calculating metrics for sample {sample.sample_id}")
        metrics = self.metrics.compute_all_metrics(response, sample.ground_truth)
        
        # LLM-as-Judge evaluation
        llm_judge_result = None
        if self.use_llm_judge and self.llm_judge:
            logger.debug(f"Running LLM judge for sample {sample.sample_id}")
            judge_result = self.llm_judge.evaluate_response(
                sample.question,
                response,
                sample.ground_truth
            )
            llm_judge_result = {
                'is_hallucinated': judge_result.is_hallucinated,
                'confidence': judge_result.confidence,
                'reasoning': judge_result.reasoning,
                'category': judge_result.category,
                'score': judge_result.score
            }
        
        return {
            'sample_id': sample.sample_id,
            'question': sample.question,
            'response': response,
            'ground_truth': sample.ground_truth,
            'metrics': metrics,
            'llm_judge': llm_judge_result,
            'status': 'success'
        }
    
    def _save_intermediate_results(self):
        """Save intermediate results to file."""
        filepath = os.path.join(self.output_dir, 'intermediate_results.json')
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump({
                'results': self.results,
                'timestamp': datetime.now().isoformat()
            }, f, indent=2, default=str)
    
    def _generate_report(self, dataset_name: str = 'benchmark') -> Dict[str, Any]:
        """Generate final benchmark report."""
        report_gen = ReportGenerator(output_dir=self.output_dir)
        # Get model name from model interface
        model_name = getattr(self.model, 'model', 'unknown')
        
        # Get usage stats if available
        usage_stats = getattr(self.model, 'get_usage_stats', lambda: {})()
        
        # Convert results to proper format for report generator
        formatted_results = []
        for r in self.results:
            if 'error' in r:
                continue  # Skip failed samples
            
            metrics = r.get('metrics', {})
            llm_judge = r.get('llm_judge', {}) or {}
            
            formatted_results.append({
                'sample_id': r.get('sample_id', 'unknown'),
                'question': r.get('question', ''),
                'response': r.get('response', ''),
                'ground_truth': r.get('ground_truth', ''),
                'faithfulness': metrics.get('faithfulness', 0.0),
                'factual_consistency': metrics.get('factual_consistency', 0.0),
                'hallucination_rate': metrics.get('hallucination_rate', 0.0),
                'bleu_score': metrics.get('bleu_score', 0.0),
                'claim_count': metrics.get('claim_count', 0),
                'citation_count': metrics.get('citation_count', 0),
                'llm_judge_score': llm_judge.get('score'),
                'llm_judge_reasoning': llm_judge.get('reasoning'),
                'hallucination_detected': llm_judge.get('is_hallucinated', False) if llm_judge else False,
                'hallucination_category': llm_judge.get('category') if llm_judge else None
            })
        
        failed_count = sum(1 for r in self.results if 'error' in r)

        report = report_gen.generate_report(
            results=formatted_results,
            model_name=model_name,
            dataset_name=dataset_name,
            config={
                'total_samples': len(self.results),
                'successful_samples': len(self.results) - failed_count,
                'failed_samples': failed_count,
                'usage_stats': usage_stats
            }
        )

        # Save primary report files with stable names used by the CLI
        report.to_json(os.path.join(self.output_dir, 'report.json'))
        report_gen._export_csv(report, os.path.join(self.output_dir, 'results.csv'))
        report_gen._export_markdown(report, os.path.join(self.output_dir, 'report.md'))

        return report.to_dict()
    
    def get_results(self) -> List[Dict[str, Any]]:
        """Get current results."""
        return self.results
