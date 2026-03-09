"""
Model interface for generating responses from target LLMs.
Supports OpenRouter API for accessing various models.
"""

import os
import time
from typing import Dict, List, Any, Optional, Generator
import requests
from abc import ABC, abstractmethod
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class ModelInterface(ABC):
    """Abstract base class for model interfaces."""
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate a response for the given prompt."""
        pass
    
    @abstractmethod
    def generate_batch(self, prompts: List[str], **kwargs) -> List[str]:
        """Generate responses for a batch of prompts."""
        pass


class OpenRouterModel(ModelInterface):
    """Interface for models via OpenRouter API."""
    
    # Pricing per 1M tokens (input/output) for common models
    PRICING = {
        # Anthropic models
        "anthropic/claude-opus-4-6": {"input": 15.00, "output": 75.00, "cache_read": 1.50, "cache_write": 18.75},
        "anthropic/claude-sonnet-4-6": {"input": 3.00, "output": 15.00, "cache_read": 0.30, "cache_write": 3.75},
        "anthropic/claude-haiku-4-5": {"input": 0.80, "output": 4.00, "cache_read": 0.08, "cache_write": 1.00},
        # Google models
        "google/gemini-3-pro-preview": {"input": 1.25, "output": 5.00, "cache_read": 0.31, "cache_write": 1.25},
        "google/gemini-3-flash-preview": {"input": 0.15, "output": 0.60, "cache_read": 0.04, "cache_write": 0.15},
        "google/gemini-2.5-pro": {"input": 1.25, "output": 10.00, "cache_read": 0.31, "cache_write": 1.25},
        "google/gemini-2.5-flash": {"input": 0.15, "output": 0.60, "cache_read": 0.04, "cache_write": 0.15},
        # OpenAI models
        "openai/gpt-5.4-pro": {"input": 30.00, "output": 180.00, "cache_read": 7.50, "cache_write": 30.00},
        "openai/gpt-5.4": {"input": 2.50, "output": 15.00, "cache_read": 1.25, "cache_write": 2.50},
        "openai/gpt-5": {"input": 10.00, "output": 30.00, "cache_read": 2.50, "cache_write": 10.00},
        "openai/o3": {"input": 10.00, "output": 40.00, "cache_read": 2.50, "cache_write": 10.00},
        "openai/o4-mini": {"input": 1.10, "output": 4.40, "cache_read": 0.28, "cache_write": 1.10},
        # Meta models
        "meta-llama/llama-4-maverick": {"input": 0.22, "output": 0.88, "cache_read": 0.00, "cache_write": 0.00},
        "meta-llama/llama-4-scout": {"input": 0.11, "output": 0.34, "cache_read": 0.00, "cache_write": 0.00},
        # Mistral models
        "mistralai/mistral-large-2512": {"input": 2.00, "output": 6.00, "cache_read": 0.00, "cache_write": 0.00},
        "mistralai/mistral-small-3.1-24b-instruct": {"input": 0.10, "output": 0.30, "cache_read": 0.00, "cache_write": 0.00},
        "mistralai/devstral-medium": {"input": 0.40, "output": 1.20, "cache_read": 0.00, "cache_write": 0.00},
    }
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "google/gemini-3-flash-preview",
        base_url: str = "https://openrouter.ai/api/v1",
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        """
        Initialize OpenRouter model interface.
        
        Args:
            api_key: OpenRouter API key
            model: Model identifier
            base_url: OpenRouter API base URL
            max_retries: Maximum number of retries on failure
            retry_delay: Delay between retries in seconds
        """
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        # Strip "openrouter/" prefix if present — OpenRouter API expects provider/model format
        self.model = model.removeprefix("openrouter/")
        self.base_url = base_url
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Usage tracking
        self.usage_stats = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cache_read_tokens": 0,
            "cache_write_tokens": 0,
            "total_cost": 0.0,
            "request_count": 0
        }
        
        if not self.api_key:
            raise ValueError("OpenRouter API key required. Set OPENROUTER_API_KEY environment variable.")
    
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate a response for the given prompt.
        
        Args:
            prompt: Input prompt
            **kwargs: Additional generation parameters
            
        Returns:
            Generated response text
        """
        temperature = kwargs.get('temperature', 0.7)
        max_tokens = kwargs.get('max_tokens', 1024)
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=data,
                    timeout=60
                )
                response.raise_for_status()
                result = response.json()
                
                # Track token usage
                self._track_usage(result)
                
                return result['choices'][0]['message']['content']
            except requests.exceptions.RequestException as e:
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    raise RuntimeError(f"Failed to generate response after {self.max_retries} attempts: {e}")
    
    def _track_usage(self, result: Dict[str, Any]):
        """Track token usage from API response."""
        usage = result.get('usage', {})
        
        prompt_tokens = usage.get('prompt_tokens', 0)
        completion_tokens = usage.get('completion_tokens', 0)
        total_tokens = usage.get('total_tokens', prompt_tokens + completion_tokens)
        
        # OpenRouter specific fields
        cache_read = usage.get('cache_read_tokens', 0) or usage.get('cache_read_input_tokens', 0)
        cache_write = usage.get('cache_write_tokens', 0) or usage.get('cache_creation_input_tokens', 0)
        
        self.usage_stats['prompt_tokens'] += prompt_tokens
        self.usage_stats['completion_tokens'] += completion_tokens
        self.usage_stats['total_tokens'] += total_tokens
        self.usage_stats['cache_read_tokens'] += cache_read
        self.usage_stats['cache_write_tokens'] += cache_write
        self.usage_stats['request_count'] += 1
        
        # Calculate cost
        cost = self._calculate_cost(prompt_tokens, completion_tokens, cache_read, cache_write)
        self.usage_stats['total_cost'] += cost
    
    def _calculate_cost(self, prompt_tokens: int, completion_tokens: int, 
                        cache_read: int = 0, cache_write: int = 0) -> float:
        """Calculate cost for a request based on token usage."""
        pricing = self.PRICING.get(self.model, {"input": 0.0, "output": 0.0, "cache_read": 0.0, "cache_write": 0.0})
        
        # Cost per 1M tokens
        input_cost = (prompt_tokens / 1_000_000) * pricing.get('input', 0)
        output_cost = (completion_tokens / 1_000_000) * pricing.get('output', 0)
        cache_read_cost = (cache_read / 1_000_000) * pricing.get('cache_read', 0)
        cache_write_cost = (cache_write / 1_000_000) * pricing.get('cache_write', 0)
        
        return input_cost + output_cost + cache_read_cost + cache_write_cost
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get current usage statistics."""
        return self.usage_stats.copy()
    
    def reset_usage_stats(self):
        """Reset usage statistics."""
        self.usage_stats = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cache_read_tokens": 0,
            "cache_write_tokens": 0,
            "total_cost": 0.0,
            "request_count": 0
        }
    
    def generate_batch(self, prompts: List[str], **kwargs) -> List[str]:
        """
        Generate responses for a batch of prompts.
        Processes sequentially with rate limiting.
        
        Args:
            prompts: List of input prompts
            **kwargs: Additional generation parameters
            
        Returns:
            List of generated responses
        """
        responses = []
        for i, prompt in enumerate(prompts):
            response = self.generate(prompt, **kwargs)
            responses.append(response)
            # Rate limiting between requests
            if i < len(prompts) - 1:
                time.sleep(0.5)
        return responses
    
    def generate_for_benchmark(
        self,
        questions: List[str],
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> List[str]:
        """
        Generate responses specifically for benchmark questions.
        
        Args:
            questions: List of questions
            system_prompt: Optional system prompt to prepend
            **kwargs: Additional generation parameters
            
        Returns:
            List of responses
        """
        prompts = []
        for question in questions:
            if system_prompt:
                prompt = f"{system_prompt}\n\nQuestion: {question}\nAnswer:"
            else:
                prompt = f"Question: {question}\nAnswer:"
            prompts.append(prompt)
        
        return self.generate_batch(prompts, **kwargs)


class MockModel(ModelInterface):
    """Mock model for testing without API calls."""
    
    def __init__(self, responses: Optional[List[str]] = None):
        """
        Initialize mock model.
        
        Args:
            responses: Predefined responses to return
        """
        self.responses = responses or ["This is a mock response."]
        self.call_count = 0
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Return mock response."""
        response = self.responses[self.call_count % len(self.responses)]
        self.call_count += 1
        return response
    
    def generate_batch(self, prompts: List[str], **kwargs) -> List[str]:
        """Return mock responses for batch."""
        return [self.generate(p) for p in prompts]
