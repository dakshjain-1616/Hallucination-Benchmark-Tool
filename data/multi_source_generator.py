"""
Multi-source consensus dataset generator.
Queries 3 distinct LLM sources via OpenRouter to cross-verify information before finalization.
Supports both factual topics and abstract parameters/traits (e.g., 'friendliness', 'technical depth').
"""

import os
import json
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.model_interface import OpenRouterModel


@dataclass
class ConsensusResult:
    """Result from multi-source consensus generation."""
    question: str
    ground_truth: str
    sources: List[Dict[str, Any]]
    consensus_score: float
    agreement_level: str  # 'high', 'medium', 'low'
    parameter_type: str  # 'factual' or 'abstract'
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'question': self.question,
            'ground_truth': self.ground_truth,
            'sources': self.sources,
            'consensus_score': self.consensus_score,
            'agreement_level': self.agreement_level,
            'parameter_type': self.parameter_type
        }


class MultiSourceGenerator:
    """
    Generates dataset entries using multi-source consensus.
    Queries 3 distinct models and synthesizes verified ground truth.
    Supports both factual topics and abstract parameters/traits.
    """
    
    # Three distinct models for cross-verification (different providers = reduced bias)
    # Updated to latest state-of-the-art models as of March 2026
    SOURCE_MODELS = [
        "openai/gpt-5.4",
        "anthropic/claude-sonnet-4-6",
        "google/gemini-3-pro-preview"
    ]
    
    # Abstract parameters that require different prompting strategy
    ABSTRACT_PARAMETERS = {
        'friendliness', 'friendliness_level', 'friendliness level',
        'technical_depth', 'technical depth',
        'creativity', 'creativity_level', 'creativity level',
        'empathy', 'empathy_level', 'empathy level',
        'professionalism', 'professionalism_level', 'professionalism level',
        'humor', 'humor_level', 'humor level',
        'clarity', 'clarity_level', 'clarity level',
        'helpfulness', 'helpfulness_level', 'helpfulness level',
        'tone', 'style', 'personality', 'behavior', 'manner'
    }
    
    def __init__(self, api_key: Optional[str] = None, temperature: float = 0.3):
        """
        Initialize the multi-source generator.
        
        Args:
            api_key: OpenRouter API key
            temperature: Temperature for generation (lower for more factual consistency)
        """
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.temperature = temperature
        
        if not self.api_key:
            raise ValueError("OpenRouter API key required. Set OPENROUTER_API_KEY environment variable.")
        
        # Initialize model interfaces for each source
        self.models = {}
        for model_id in self.SOURCE_MODELS:
            self.models[model_id] = OpenRouterModel(
                api_key=self.api_key,
                model=model_id,
                max_retries=2,
                retry_delay=1.0
            )
    
    def _is_abstract_parameter(self, topic: str) -> bool:
        """Check if the topic is an abstract parameter/trait rather than a factual topic."""
        topic_lower = topic.lower().strip()
        # Check exact match or if topic contains any abstract parameter keywords
        return (
            topic_lower in self.ABSTRACT_PARAMETERS or
            any(param in topic_lower for param in self.ABSTRACT_PARAMETERS)
        )
    
    def _calculate_agreement(self, responses: List[Dict[str, Any]]) -> float:
        """
        Calculate agreement score between multiple responses.
        Uses simple text similarity based on shared words.
        
        Args:
            responses: List of response dictionaries with 'answer' key
            
        Returns:
            Agreement score between 0.0 and 1.0
        """
        if len(responses) < 2:
            return 1.0 if len(responses) == 1 else 0.0
        
        from difflib import SequenceMatcher
        
        answers = [r['answer'].lower().strip() for r in responses if 'answer' in r]
        if len(answers) < 2:
            return 0.0
        
        # Calculate pairwise similarity
        similarities = []
        for i in range(len(answers)):
            for j in range(i + 1, len(answers)):
                similarity = SequenceMatcher(None, answers[i], answers[j]).ratio()
                similarities.append(similarity)
        
        # Return average similarity
        return sum(similarities) / len(similarities) if similarities else 0.0
    
    def _generate_abstract_ground_truth(self, parameter: str, question: str, source_responses: List[Dict[str, Any]]) -> Tuple[str, float, str]:
        """
        Synthesize consensus ground truth for abstract parameters.
        Uses a judge model to determine the best response based on the parameter being evaluated.
        """
        # Filter out error responses
        valid_responses = [r for r in source_responses if 'error' not in r]
        
        if len(valid_responses) < 2:
            answer = valid_responses[0]['answer'] if valid_responses else "[No valid responses]"
            return answer, 0.0, "low"
        
        # Use Claude as judge for abstract parameters
        judge_model = self.models[self.SOURCE_MODELS[1]]
        
        # Build comparison prompt for abstract evaluation
        answers_text = "\n\n".join([
            f"RESPONSE {i+1} ({r['model']}):\n{r['answer']}"
            for i, r in enumerate(valid_responses)
        ])
        
        consensus_prompt = f"""You are evaluating multiple AI responses to determine the best example of {parameter}.

Scenario: {question}

{answers_text}

Your task:
1. Evaluate each response for its demonstration of {parameter}
2. Identify which response best exemplifies this trait
3. Synthesize the key characteristics that make a response strong in {parameter}
4. Create an "ideal" reference response that demonstrates {parameter} effectively

Provide ONLY the final ideal reference response, no explanation.

Ideal Response:"""
        
        consensus_answer = judge_model.generate(
            consensus_prompt, 
            temperature=0.1, 
            max_tokens=400
        ).strip()
        
        # Calculate agreement based on response similarity for abstract parameters
        agreement_score = self._calculate_agreement(valid_responses)
        
        if agreement_score >= 0.7:  # Slightly lower threshold for abstract
            agreement_level = "high"
        elif agreement_score >= 0.4:
            agreement_level = "medium"
        else:
            agreement_level = "low"
        
        return consensus_answer, agreement_score, agreement_level
    
    def _query_all_sources(self, question: str, is_abstract: bool = False) -> List[Dict[str, Any]]:
        """Query all 3 sources for answers to the question."""
        responses = []
        
        if is_abstract:
            # For abstract parameters, ask for responses that demonstrate the trait
            prompt = f"""Respond to the following scenario. Provide a natural, authentic response.

Scenario: {question}

Your Response:"""
        else:
            # For factual topics, ask for concise factual answers
            prompt = f"""Answer the following question concisely and factually.
Provide ONLY the factual answer, no preamble or explanation.

Question: {question}

Answer:"""
        
        for model_id in self.SOURCE_MODELS:
            try:
                model = self.models[model_id]
                response = model.generate(prompt, temperature=self.temperature, max_tokens=300)
                
                # Get usage stats
                stats = model.get_usage_stats()
                
                responses.append({
                    'model': model_id,
                    'answer': response.strip(),
                    'prompt_tokens': stats.get('prompt_tokens', 0),
                    'completion_tokens': stats.get('completion_tokens', 0),
                    'cost': stats.get('total_cost', 0.0)
                })
                
                # Reset stats for next query
                model.reset_usage_stats()
                
            except Exception as e:
                responses.append({
                    'model': model_id,
                    'answer': f"[Error: {str(e)}]",
                    'error': str(e)
                })
        
        return responses
    
    def _synthesize_consensus(
        self, 
        question: str, 
        source_responses: List[Dict[str, Any]],
        parameter_type: str = "factual"
    ) -> Tuple[str, float, str]:
        """
        Synthesize consensus ground truth from multiple sources.
        Uses a judge model to determine the best answer.
        
        Returns:
            Tuple of (ground_truth, consensus_score, agreement_level)
        """
        # Filter out error responses
        valid_responses = [r for r in source_responses if 'error' not in r]
        
        if len(valid_responses) < 2:
            # Fall back to first valid response
            answer = valid_responses[0]['answer'] if valid_responses else "[No valid responses]"
            return answer, 0.0, "low"
        
        # Use the judge model (Claude) to synthesize consensus
        judge_model = self.models[self.SOURCE_MODELS[1]]  # Claude as judge
        
        # Build comparison prompt
        answers_text = "\n\n".join([
            f"SOURCE {i+1} ({r['model']}):\n{r['answer']}"
            for i, r in enumerate(valid_responses)
        ])
        
        if parameter_type == "abstract":
            # For abstract parameters, focus on quality of trait demonstration
            consensus_prompt = f"""You are evaluating multiple responses to determine the best example that demonstrates the intended quality.

Scenario: {question}

{answers_text}

Your task:
1. Compare the responses for quality and appropriateness
2. Identify the best elements from each response
3. Synthesize a consensus "ideal" response that represents the best qualities
4. Focus on creating a natural, authentic response

Provide ONLY the final consensus response, synthesized from the best elements of the sources.

Consensus Response:"""
        else:
            # For factual topics, focus on factual accuracy
            consensus_prompt = f"""You are evaluating multiple answers to determine the most accurate consensus.

Question: {question}

{answers_text}

Your task:
1. Compare the answers for factual accuracy and completeness
2. Identify any contradictions between sources
3. Synthesize the most accurate consensus answer
4. If sources disagree, choose the most detailed and well-supported answer

Provide ONLY the final consensus answer, synthesized from the best elements of the sources.

Consensus Answer:"""
        
        consensus_answer = judge_model.generate(
            consensus_prompt, 
            temperature=0.1, 
            max_tokens=400
        ).strip()
        
        # Calculate agreement level based on similarity
        agreement_score = self._calculate_agreement(valid_responses)
        
        # Adjust thresholds for abstract parameters
        if parameter_type == "abstract":
            if agreement_score >= 0.7:
                agreement_level = "high"
            elif agreement_score >= 0.4:
                agreement_level = "medium"
            else:
                agreement_level = "low"
        else:
            if agreement_score >= 0.8:
                agreement_level = "high"
            elif agreement_score >= 0.5:
                agreement_level = "medium"
            else:
                agreement_level = "low"
        
        return consensus_answer, agreement_score, agreement_level
    
    def generate_topic_entry(self, topic: str, entry_index: int = 0) -> ConsensusResult:
        """
        Generate a single dataset entry for a specific topic using multi-source consensus.
        
        Args:
            topic: The topic/domain (e.g., 'geography', 'maths reasoning', 'hollywood movies')
                   or abstract parameter (e.g., 'friendliness', 'technical depth')
            entry_index: Index for generating varied questions
            
        Returns:
            ConsensusResult with verified question and ground truth
        """
        # Determine if this is an abstract parameter
        parameter_type = "abstract" if self._is_abstract_parameter(topic) else "factual"
        
        # Step 1: Generate question from the topic
        question = self._generate_question(topic, entry_index, parameter_type)
        
        # Step 2: Query all 3 sources for answers
        source_responses = self._query_all_sources(question, parameter_type)
        
        # Step 3: Synthesize consensus ground truth
        ground_truth, consensus_score, agreement_level = self._synthesize_consensus(
            question, source_responses, parameter_type
        )
        
        return ConsensusResult(
            question=question,
            ground_truth=ground_truth,
            sources=source_responses,
            consensus_score=consensus_score,
            agreement_level=agreement_level,
            parameter_type=parameter_type
        )
    
    def generate_dataset(
        self, 
        topic: str, 
        num_entries: int = 5,
        output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate a complete dataset for a topic using multi-source consensus.
        
        Args:
            topic: The topic/domain for generation (factual or abstract)
            num_entries: Number of entries to generate
            output_path: Optional path to save the dataset
            
        Returns:
            Dictionary containing the dataset
        """
        parameter_type = "abstract" if self._is_abstract_parameter(topic) else "factual"
        
        print(f"\nGenerating {num_entries} entries for topic: '{topic}'")
        print(f"Parameter type: {parameter_type}")
        print("Using 3-source consensus: GPT-5.4, Claude Sonnet 4.6, Gemini 3 Pro Preview")
        print("-" * 60)
        
        samples = []
        total_cost = 0.0
        
        for i in range(num_entries):
            print(f"\n[Entry {i+1}/{num_entries}] Generating...")
            
            try:
                result = self.generate_topic_entry(topic, i)
                
                # Collect cost from all models
                entry_cost = 0.0
                for model_id in self.SOURCE_MODELS:
                    stats = self.models[model_id].get_usage_stats()
                    entry_cost += stats.get('total_cost', 0.0)
                
                total_cost += entry_cost
                
                sample = {
                    'id': f"{topic.replace(' ', '_')}_{i+1}",
                    'question': result.question,
                    'ground_truth': result.ground_truth,
                    'metadata': {
                        'topic': topic,
                        'parameter_type': result.parameter_type,
                        'consensus_score': result.consensus_score,
                        'agreement_level': result.agreement_level,
                        'sources': result.sources,
                        'generation_cost': entry_cost
                    }
                }
                samples.append(sample)
                
                print(f"  ✓ Question: {result.question[:60]}...")
                print(f"  ✓ Agreement: {result.agreement_level} (score: {result.consensus_score:.2f})")
                print(f"  ✓ Cost: ${entry_cost:.4f}")
                
                # Rate limiting between entries
                if i < num_entries - 1:
                    time.sleep(1.0)
                    
            except Exception as e:
                print(f"  ✗ Error generating entry {i+1}: {e}")
                continue
        
        dataset = {
            'name': f"{topic.replace(' ', '_')}_consensus_dataset",
            'metadata': {
                'topic': topic,
                'parameter_type': parameter_type,
                'num_entries': len(samples),
                'source_models': self.SOURCE_MODELS,
                'total_generation_cost': total_cost,
                'generation_method': 'multi_source_consensus'
            },
            'samples': samples
        }
        
        if output_path:
            os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(dataset, f, indent=2)
            print(f"\n✓ Dataset saved to: {output_path}")
        
        print(f"\n{'=' * 60}")
        print(f"Generation complete!")
        print(f"  Total entries: {len(samples)}")
        print(f"  Total cost: ${total_cost:.4f}")
        print(f"  Average cost per entry: ${total_cost/len(samples):.4f}" if samples else "  N/A")
        
        return dataset
    
    def _generate_question(self, topic: str, index: int, parameter_type: str = "factual") -> str:
        """
        Generate a question for the given topic or abstract parameter.
        
        For factual topics: Generate knowledge-based questions.
        For abstract parameters (e.g., 'friendliness'): Generate scenario-based evaluation questions.
        """
        if parameter_type == "abstract":
            prompt = self._build_abstract_prompt(topic, index)
        else:
            prompt = f"""Generate a specific, factual question about {topic}.
The question should:
- Be clear and unambiguous
- Have a definitive factual answer
- Be suitable for testing knowledge in this domain
- Vary in difficulty (question #{index + 1})

Provide ONLY the question, no preamble or explanation.

Question:"""
        
        # Use first model for question generation
        model = self.models[self.SOURCE_MODELS[0]]
        question = model.generate(prompt, temperature=0.7, max_tokens=150).strip()
        
        # Clean up the question
        question = question.replace('"', '').replace("Question:", "").replace("Scenario:", "").strip()
        if '?' not in question and parameter_type == "factual":
            question += '?'
        
        return question
    
    def _build_abstract_prompt(self, parameter: str, index: int) -> str:
        """
        Build a prompt for generating evaluation questions for abstract parameters.
        
        Examples:
        - 'friendliness': Generate scenarios testing conversational warmth
        - 'technical depth': Generate scenarios testing explanation complexity
        - 'creativity': Generate scenarios testing novel problem-solving
        """
        # Map common abstract parameters to their evaluation contexts
        parameter_contexts = {
            'friendliness': {
                'context': 'customer service or casual conversation',
                'evaluation': 'warmth, approachability, and positive tone',
                'scenario_type': 'user inquiry or greeting'
            },
            'technical_depth': {
                'context': 'technical explanation or documentation',
                'evaluation': 'level of detail, precision, and complexity',
                'scenario_type': 'technical question requiring explanation'
            },
            'creativity': {
                'context': 'problem-solving or content generation',
                'evaluation': 'novelty, originality, and imaginative thinking',
                'scenario_type': 'open-ended creative challenge'
            },
            'empathy': {
                'context': 'emotional support or counseling',
                'evaluation': 'understanding, validation, and emotional awareness',
                'scenario_type': 'user expressing difficulty or emotion'
            },
            'professionalism': {
                'context': 'business or formal communication',
                'evaluation': 'formality, respect, and business etiquette',
                'scenario_type': 'professional inquiry or request'
            },
            'clarity': {
                'context': 'instruction or explanation',
                'evaluation': 'simplicity, understandability, and lack of jargon',
                'scenario_type': 'complex topic requiring simplification'
            },
            'helpfulness': {
                'context': 'assistance or guidance',
                'evaluation': 'actionability, completeness, and usefulness',
                'scenario_type': 'user seeking help with a task'
            }
        }
        
        # Find matching context or use generic
        param_lower = parameter.lower().replace('_', ' ').replace('-', ' ')
        context = None
        for key, ctx in parameter_contexts.items():
            if key in param_lower or param_lower in key:
                context = ctx
                break
        
        if not context:
            # Generic abstract parameter prompt
            context = {
                'context': f'evaluation of {parameter}',
                'evaluation': f'demonstration of {parameter}',
                'scenario_type': 'scenario requiring demonstration of this trait'
            }
        
        prompt = f"""Generate a specific evaluation scenario to test an AI's {parameter}.

Context: {context['context']}
Evaluation Focus: {context['evaluation']}
Scenario Type: {context['scenario_type']}
Variation: Scenario #{index + 1} (vary the specific situation)

The scenario should:
- Present a clear situation or user input that requires a response
- Be designed to evaluate {parameter} specifically
- Have a clear "ideal" or "expected" response characteristic
- Be suitable for automated evaluation

Provide ONLY the scenario description (what the user says or the situation), no preamble.

Scenario:"""
        
        return prompt
    
    def _query_all_sources(self, question: str, parameter_type: str = "factual") -> List[Dict[str, Any]]:
        """Query all 3 sources for answers to the question."""
        responses = []
        
        if parameter_type == "abstract":
            # For abstract parameters, ask for natural responses to the scenario
            prompt = f"""Respond to the following scenario naturally and authentically.

Scenario: {question}

Your Response:"""
        else:
            # For factual topics, ask for concise factual answers
            prompt = f"""Answer the following question concisely and factually.
Provide ONLY the factual answer, no preamble or explanation.

Question: {question}

Answer:"""
        
        for model_id in self.SOURCE_MODELS:
            try:
                model = self.models[model_id]
                response = model.generate(prompt, temperature=self.temperature, max_tokens=300)
                
                # Get usage stats
                stats = model.get_usage_stats()
                
                responses.append({
                    'model': model_id,
                    'answer': response.strip(),
                    'prompt_tokens': stats.get('prompt_tokens', 0),
                    'completion_tokens': stats.get('completion_tokens', 0),
                    'cost': stats.get('total_cost', 0.0)
                })
                
                # Reset stats for next query
                model.reset_usage_stats()
                
            except Exception as e:
                responses.append({
                    'model': model_id,
                    'answer': f"[Error: {str(e)}]",
                    'error': str(e)
                })
        
        return responses


def validate_single_query(topic: str = "geography") -> None:
    """Validate the multi-source generator with a single topical query."""
    print("=" * 60)
    print("Multi-Source Consensus Generator - Validation Test")
    print("=" * 60)
    print(f"\nTesting with topic: '{topic}'")
    
    generator = MultiSourceGenerator()
    result = generator.generate_topic_entry(topic, 0)
    
    print("\n" + "=" * 60)
    print("RESULT")
    print("=" * 60)
    print(f"\nQuestion: {result.question}")
    print(f"\nConsensus Ground Truth:\n{result.ground_truth}")
    print(f"\nAgreement Level: {result.agreement_level}")
    print(f"Consensus Score: {result.consensus_score:.2f}")
    print(f"Parameter Type: {result.parameter_type}")
    
    print("\n" + "-" * 60)
    print("Source Responses:")
    print("-" * 60)
    for source in result.sources:
        print(f"\n{source['model']}:")
        print(f"  Answer: {source.get('answer', '[Error]')}")
        if 'cost' in source:
            print(f"  Cost: ${source['cost']:.4f}")


if __name__ == "__main__":
    # Run validation test
    import sys
    if len(sys.argv) > 1:
        validate_single_query(sys.argv[1])
    else:
        validate_single_query("geography")
