"""
LLM-as-Judge module for semantic evaluation of hallucinations.
Uses OpenRouter API for evaluation.
"""

import os
import json
import re
from typing import Dict, List, Any, Optional
import requests
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


@dataclass
class JudgeResult:
    """Result from LLM judge evaluation."""
    is_hallucinated: bool
    confidence: float
    reasoning: str
    category: str  # 'contradictory', 'fabricated_citation', 'unsupported_claim', 'contextual_hallucination', 'none'
    score: float  # 0-1 scale
    hallucination_types: Dict[str, int] = None  # Breakdown by type
    
    def __post_init__(self):
        if self.hallucination_types is None:
            self.hallucination_types = {}


class LLMJudge:
    """LLM-as-Judge for hallucination detection with few-shot examples."""
    
    # Few-shot examples for the judge
    FEW_SHOT_EXAMPLES = """
EXAMPLE 1 - CONTRADICTORY HALLUCINATION:
Question: What is the boiling point of water at sea level?
Ground Truth: Water boils at 100 degrees Celsius (212 degrees Fahrenheit) at sea level.
Model Response: Water boils at 90 degrees Celsius at sea level due to atmospheric pressure variations.

Analysis: This is a CONTRADICTORY hallucination. The response states 90°C which directly contradicts the established fact of 100°C. The explanation about atmospheric pressure is used to justify the false claim.
Classification: contradictory
Severity: High (1.0)

---

EXAMPLE 2 - FABRICATED CITATION:
Question: What are the health benefits of meditation?
Ground Truth: Meditation has been shown to reduce stress, improve focus, and decrease anxiety in various studies.
Model Response: According to a 2019 study by Dr. Sarah Johnson et al. published in the Journal of Neuroscience, meditation increases brain gray matter by 25% after just 2 weeks of practice [1].

Analysis: This contains a FABRICATED CITATION. The specific study by "Dr. Sarah Johnson et al." with the exact claim of "25% increase in gray matter after 2 weeks" appears fabricated. The citation format [1] suggests a reference list that doesn't exist.
Classification: fabricated_citation
Severity: High (0.9)

---

EXAMPLE 3 - UNSUPPORTED CLAIM:
Question: Who invented the telephone?
Ground Truth: Alexander Graham Bell is credited with inventing the first practical telephone in 1876.
Model Response: Alexander Graham Bell invented the telephone in 1876. He was also a talented pianist and spoke five languages fluently. The telephone patent was filed on March 7, 1876.

Analysis: This contains UNSUPPORTED CLAIMS. While the core fact about Bell inventing the telephone in 1876 is correct, the additional details about him being a "talented pianist" and speaking "five languages fluently" are not found in the ground truth. These are not contradictory but are unsupported additions.
Classification: unsupported_claim
Severity: Moderate (0.6)

---

EXAMPLE 4 - CONTEXTUAL HALLUCINATION (Technically True but Irrelevant):
Question: What is the capital of France?
Ground Truth: The capital of France is Paris.
Model Response: The capital of France is Paris. Paris is also home to the Eiffel Tower, which was completed in 1889 and stands 330 meters tall. The tower was designed by Gustave Eiffel.

Analysis: This is a CONTEXTUAL HALLUCINATION. While the Eiffel Tower facts are technically true in the real world, they are NOT present in the provided ground truth context. The model is introducing external information that, while factually accurate generally, is unsupported by the specific context provided.
Classification: contextual_hallucination
Severity: Low-Moderate (0.4)

---

EXAMPLE 5 - CORRECT (No Hallucination):
Question: What is photosynthesis?
Ground Truth: Photosynthesis is the process by which plants use sunlight, water, and carbon dioxide to create oxygen and energy in the form of sugar.
Model Response: Photosynthesis is the process where plants convert sunlight, water, and carbon dioxide into oxygen and glucose (sugar), producing energy for the plant.

Analysis: This response is CORRECT with no hallucination. All key elements from the ground truth are present: sunlight, water, carbon dioxide, oxygen, sugar/glucose, and the process of energy production. The wording differs but the factual content matches.
Classification: none
Severity: None (1.0 - perfect score)
"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "google/gemini-2.0-flash-001"):
        """
        Initialize the LLM Judge.
        
        Args:
            api_key: OpenRouter API key (defaults to OPENROUTER_API_KEY env var)
            model: Model identifier for OpenRouter
        """
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        # Strip "openrouter/" prefix if present — OpenRouter API expects provider/model format
        self.model = model.removeprefix("openrouter/")
        self.base_url = "https://openrouter.ai/api/v1"
        
        if not self.api_key:
            raise ValueError("OpenRouter API key required. Set OPENROUTER_API_KEY environment variable.")
    
    def evaluate_response(self, question: str, response: str, ground_truth: str) -> JudgeResult:
        """
        Evaluate a response for hallucinations using LLM-as-Judge.
        
        Args:
            question: The original question
            response: The model's response
            ground_truth: The ground truth answer
            
        Returns:
            JudgeResult with evaluation details
        """
        prompt = self._build_evaluation_prompt(question, response, ground_truth)
        
        try:
            result = self._call_llm(prompt)
            return self._parse_evaluation(result)
        except Exception as e:
            # Return conservative result on error
            return JudgeResult(
                is_hallucinated=True,
                confidence=0.0,
                reasoning=f"Evaluation error: {str(e)}",
                category="unknown",
                score=0.0,
                hallucination_types={}
            )
    
    def evaluate_batch(self, items: List[Dict[str, str]]) -> List[JudgeResult]:
        """Evaluate a batch of responses."""
        results = []
        for item in items:
            result = self.evaluate_response(
                item['question'],
                item['response'],
                item['ground_truth']
            )
            results.append(result)
        return results
    
    def _build_evaluation_prompt(self, question: str, response: str, ground_truth: str) -> str:
        """Build the evaluation prompt for the LLM judge with few-shot examples."""
        return f"""You are an expert evaluator assessing whether a language model's response contains hallucinations.

A hallucination is defined as:
1. **Contradictory** - Statements that directly contradict the ground truth (most severe)
2. **Fabricated citations** - References to non-existent sources like fake studies, papers, or authors (very severe)
3. **Unsupported claims** - Assertions not found in the ground truth but not contradictory (moderate)
4. **Contextual hallucinations** - Information that may be true in general but is NOT in the provided ground truth context (less severe but still penalized)

IMPORTANT: Penalize claims that are technically true in the real world but NOT present in the provided ground truth context. The model should only use information from the ground truth.

{self.FEW_SHOT_EXAMPLES}

---

NOW EVALUATE THIS RESPONSE:

QUESTION:
{question}

MODEL RESPONSE:
{response}

GROUND TRUTH:
{ground_truth}

Provide your assessment in the following JSON format:
{{
    "is_hallucinated": true/false,
    "confidence": 0.0-1.0,
    "reasoning": "detailed explanation of findings, citing specific parts of the response",
    "category": "contradictory|fabricated_citation|unsupported_claim|contextual_hallucination|none",
    "score": 0.0-1.0,  // overall score (1.0 = perfect, 0.0 = complete hallucination)
    "hallucination_types": {{
        "contradictory": 0,  // count of contradictory claims
        "fabricated_citation": 0,  // count of fabricated citations
        "unsupported_claim": 0,  // count of unsupported claims
        "contextual_hallucination": 0  // count of contextual hallucinations
    }}
}}

Be thorough and objective. Identify ALL types of hallucinations present, not just the primary one."""
    
    def _call_llm(self, prompt: str) -> str:
        """Call the OpenRouter API."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1,
            "max_tokens": 1500
        }
        
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=data,
            timeout=60
        )
        
        response.raise_for_status()
        result = response.json()
        
        return result['choices'][0]['message']['content']
    
    def _parse_evaluation(self, text: str) -> JudgeResult:
        """Parse the LLM evaluation response."""
        try:
            # Extract JSON from response — use brace-counting to handle nested objects
            json_str = None
            start = text.find('{')
            if start != -1:
                depth = 0
                for i, ch in enumerate(text[start:], start):
                    if ch == '{':
                        depth += 1
                    elif ch == '}':
                        depth -= 1
                        if depth == 0:
                            json_str = text[start:i + 1]
                            break
            if json_str:
                # Fix common JSON issues
                json_str = re.sub(r',\s*}', '}', json_str)
                data = json.loads(json_str)
            else:
                # Try parsing entire response
                data = json.loads(text)
            
            # Extract hallucination type counts
            hallucination_types = data.get('hallucination_types', {})
            
            return JudgeResult(
                is_hallucinated=data.get('is_hallucinated', False),
                confidence=float(data.get('confidence', 0.5)),
                reasoning=data.get('reasoning', 'No reasoning provided'),
                category=data.get('category', 'unknown'),
                score=float(data.get('score', 0.5)),
                hallucination_types=hallucination_types
            )
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            # Fallback: parse from text
            return self._fallback_parse(text)
    
    def _fallback_parse(self, text: str) -> JudgeResult:
        """Fallback parsing when JSON extraction fails."""
        text_lower = text.lower()
        
        # Determine if hallucinated based on keywords
        is_hallucinated = any(word in text_lower for word in ['hallucinat', 'incorrect', 'error', 'fabricated', 'false', 'contradict'])
        
        # Try to extract category
        category = 'none'
        if 'contradict' in text_lower:
            category = 'contradictory'
        elif 'citation' in text_lower:
            category = 'fabricated_citation'
        elif 'unsupported' in text_lower:
            category = 'unsupported_claim'
        elif 'contextual' in text_lower:
            category = 'contextual_hallucination'
        
        return JudgeResult(
            is_hallucinated=is_hallucinated,
            confidence=0.5,
            reasoning=text[:500],  # Truncate for brevity
            category=category,
            score=0.0 if is_hallucinated else 1.0,
            hallucination_types={}
        )
