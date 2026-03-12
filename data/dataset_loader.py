"""
Dataset loader module for hallucination benchmark.
Handles loading, validation, and preparation of question-answer-ground truth data.
"""

import json
import csv
import os
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, asdict
import pandas as pd


@dataclass
class BenchmarkSample:
    """Single sample in the benchmark dataset."""
    question: str
    ground_truth: str
    response: Optional[str] = None
    sample_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BenchmarkDataset:
    """Container for benchmark dataset."""
    samples: List[BenchmarkSample]
    name: str = "benchmark"
    metadata: Optional[Dict[str, Any]] = None
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __iter__(self):
        return iter(self.samples)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'metadata': self.metadata,
            'samples': [s.to_dict() for s in self.samples]
        }


class DatasetLoader:
    """Load and validate benchmark datasets."""
    
    SUPPORTED_FORMATS = ['.json', '.jsonl', '.csv', '.parquet']
    
    def __init__(self, data_dir: str = "./data"):
        """
        Initialize the dataset loader.
        
        Args:
            data_dir: Directory containing datasets
        """
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
    
    def load(self, filepath: str) -> BenchmarkDataset:
        """
        Load a dataset from file.
        
        Args:
            filepath: Path to dataset file
            
        Returns:
            BenchmarkDataset object
            
        Raises:
            ValueError: If file format is not supported
            FileNotFoundError: If file does not exist
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Dataset file not found: {filepath}")
        
        ext = os.path.splitext(filepath)[1].lower()
        
        if ext == '.json':
            return self._load_json(filepath)
        elif ext == '.jsonl':
            return self._load_jsonl(filepath)
        elif ext == '.csv':
            return self._load_csv(filepath)
        elif ext == '.parquet':
            return self._load_parquet(filepath)
        else:
            raise ValueError(f"Unsupported file format: {ext}. Supported: {self.SUPPORTED_FORMATS}")
    
    def _load_json(self, filepath: str) -> BenchmarkDataset:
        """Load JSON dataset."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return self._parse_data(data)
    
    def _load_jsonl(self, filepath: str) -> BenchmarkDataset:
        """Load JSONL dataset."""
        samples = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(json.loads(line))
        
        return self._parse_data({'samples': samples})
    
    def _load_csv(self, filepath: str) -> BenchmarkDataset:
        """Load CSV dataset."""
        df = pd.read_csv(filepath)
        return self._parse_dataframe(df)
    
    def _load_parquet(self, filepath: str) -> BenchmarkDataset:
        """Load Parquet dataset."""
        df = pd.read_parquet(filepath)
        return self._parse_dataframe(df)
    
    def _parse_data(self, data: Dict[str, Any]) -> BenchmarkDataset:
        """Parse dictionary data into BenchmarkDataset."""
        name = data.get('name', 'benchmark')
        metadata = data.get('metadata', {})
        samples_data = data.get('samples', [])
        
        samples = []
        for i, sample_data in enumerate(samples_data):
            sample = BenchmarkSample(
                question=sample_data.get('question', ''),
                ground_truth=sample_data.get('ground_truth', sample_data.get('answer', '')),
                response=sample_data.get('response'),
                sample_id=sample_data.get('id', f"sample_{i}"),
                metadata=sample_data.get('metadata')
            )
            samples.append(sample)
        
        return BenchmarkDataset(samples=samples, name=name, metadata=metadata)
    
    def _parse_dataframe(self, df: pd.DataFrame) -> BenchmarkDataset:
        """Parse DataFrame into BenchmarkDataset."""
        samples = []
        for i, row in df.iterrows():
            sample = BenchmarkSample(
                question=row.get('question', ''),
                ground_truth=row.get('ground_truth', row.get('answer', '')),
                response=row.get('response') if 'response' in row else None,
                sample_id=row.get('id', f"sample_{i}"),
                metadata={k: v for k, v in row.items() if k not in ['question', 'ground_truth', 'answer', 'response', 'id']}
            )
            samples.append(sample)
        
        return BenchmarkDataset(samples=samples, name="benchmark")
    
    def load_from_dict(self, data: Dict[str, Any]) -> BenchmarkDataset:
        """
        Load a dataset from a dictionary (for in-memory datasets).
        
        Args:
            data: Dictionary containing dataset with 'samples', 'name', 'metadata'
            
        Returns:
            BenchmarkDataset object
        """
        return self._parse_data(data)
    
    def create_sample_dataset(self, filepath: str, format: str = 'json') -> str:
        """
        Create a sample dataset for testing.
        
        Args:
            filepath: Output file path
            format: Output format ('json', 'jsonl', 'csv')
            
        Returns:
            Path to created file
        """
        sample_data = {
            "name": "sample_benchmark",
            "metadata": {"description": "Sample dataset for testing"},
            "samples": [
                {
                    "id": "sample_1",
                    "question": "What is the capital of France?",
                    "ground_truth": "The capital of France is Paris.",
                    "metadata": {"category": "geography"}
                },
                {
                    "id": "sample_2",
                    "question": "Who wrote 'Romeo and Juliet'?",
                    "ground_truth": "William Shakespeare wrote 'Romeo and Juliet'.",
                    "metadata": {"category": "literature"}
                },
                {
                    "id": "sample_3",
                    "question": "What is the speed of light?",
                    "ground_truth": "The speed of light in vacuum is approximately 299,792,458 meters per second.",
                    "metadata": {"category": "physics"}
                }
            ]
        }
        
        os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
        
        if format == 'json':
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(sample_data, f, indent=2)
        elif format == 'jsonl':
            with open(filepath, 'w', encoding='utf-8') as f:
                for sample in sample_data['samples']:
                    f.write(json.dumps(sample) + '\n')
        elif format == 'csv':
            df = pd.DataFrame(sample_data['samples'])
            df.to_csv(filepath, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        return filepath
