"""
Metrics module for hallucination detection.
Implements faithfulness, factual consistency, and granular hallucination rate calculations.
"""

import re
import string
from typing import Dict, List, Tuple, Any, Optional
from collections import Counter
import numpy as np


class HallucinationType:
    """Enumeration of hallucination types with severity weights."""
    CONTRADICTORY = "contradictory"          # Direct contradiction with ground truth
    FABRICATED_CITATION = "fabricated_citation"  # Made-up references/sources
    UNSUPPORTED_CLAIM = "unsupported_claim"    # Claims not in ground truth (not contradictory)
    CONTEXTUAL_HALLUCINATION = "contextual_hallucination"  # True but irrelevant to context
    NONE = "none"


class HallucinationMetrics:
    """Calculate hallucination-related metrics for LLM responses with granular classification."""
    
    # Severity weights for different hallucination types (higher = more severe)
    HALLUCINATION_WEIGHTS = {
        HallucinationType.CONTRADICTORY: 1.0,           # Most severe - directly wrong
        HallucinationType.FABRICATED_CITATION: 0.9,     # Very severe - fake sources
        HallucinationType.UNSUPPORTED_CLAIM: 0.6,       # Moderate - unverified info
        HallucinationType.CONTEXTUAL_HALLUCINATION: 0.4,  # Less severe - off-topic but true
        HallucinationType.NONE: 0.0
    }
    
    def __init__(self):
        self.metrics_history = []
    
    def calculate_faithfulness(self, response: str, ground_truth: str) -> float:
        """
        Calculate faithfulness score based on n-gram overlap.
        Measures how well the response aligns with ground truth facts.
        """
        response_tokens = self._tokenize(response.lower())
        ground_tokens = self._tokenize(ground_truth.lower())
        
        if not ground_tokens:
            return 0.0
        
        # Calculate unigram precision
        response_counter = Counter(response_tokens)
        ground_counter = Counter(ground_tokens)
        
        overlap = sum((response_counter & ground_counter).values())
        total_response = sum(response_counter.values())
        
        if total_response == 0:
            return 0.0
        
        precision = overlap / total_response
        return round(precision, 4)
    
    def calculate_factual_consistency(self, claims: List[str], ground_truth: str) -> float:
        """
        Calculate factual consistency by checking claim coverage in ground truth.
        Returns percentage of claims supported by ground truth.
        """
        if not claims:
            return 1.0
        
        ground_truth_lower = ground_truth.lower()
        supported = 0
        
        for claim in claims:
            claim_lower = claim.lower()
            # Check for exact substring match or high overlap
            if self._claim_supported(claim_lower, ground_truth_lower):
                supported += 1
        
        return round(supported / len(claims), 4)
    
    def classify_claims(self, claims: List[str], response: str, 
                        ground_truth: str) -> Dict[str, Any]:
        """
        Classify each claim into hallucination types.
        
        Returns:
            Dictionary with classification results and detailed breakdown.
        """
        ground_truth_lower = ground_truth.lower()
        classifications = []
        
        # Track citations for fabricated citation detection
        citations = self.detect_fabricated_citations(response)
        citation_positions = [c['position'] for c in citations]
        
        for claim in claims:
            claim_lower = claim.lower()
            claim_classification = self._classify_single_claim(
                claim, claim_lower, ground_truth_lower, response, citation_positions
            )
            classifications.append(claim_classification)
        
        # Calculate weighted hallucination rate
        weighted_rate = self._calculate_weighted_hallucination_rate(classifications)
        
        # Count by category
        category_counts = {}
        for c in classifications:
            cat = c['category']
            category_counts[cat] = category_counts.get(cat, 0) + 1
        
        return {
            'classifications': classifications,
            'weighted_hallucination_rate': weighted_rate,
            'category_counts': category_counts,
            'total_claims': len(claims)
        }
    
    def _classify_single_claim(self, claim: str, claim_lower: str, 
                               ground_truth_lower: str, response: str,
                               citation_positions: List[Tuple[int, int]]) -> Dict[str, Any]:
        """Classify a single claim into a hallucination type."""
        
        # Check if claim is directly supported
        if self._claim_supported(claim_lower, ground_truth_lower):
            return {
                'claim': claim,
                'category': HallucinationType.NONE,
                'weight': self.HALLUCINATION_WEIGHTS[HallucinationType.NONE],
                'reason': 'Claim is supported by ground truth'
            }
        
        # Check for direct contradiction
        contradiction_indicators = ['not', 'never', 'no ', 'false', 'incorrect', 'wrong']
        has_negation = any(ind in claim_lower for ind in contradiction_indicators)
        
        # Check if claim contradicts ground truth (opposite of what's in ground truth)
        ground_claims = self.extract_claims(ground_truth_lower)
        is_contradictory = False
        
        for gc in ground_claims:
            gc_lower = gc.lower()
            # Check for negation patterns
            if has_negation and self._has_significant_overlap(claim_lower.replace('not ', ''), gc_lower):
                is_contradictory = True
                break
            # Check for direct opposite statements
            if self._is_opposite_claim(claim_lower, gc_lower):
                is_contradictory = True
                break
        
        if is_contradictory:
            return {
                'claim': claim,
                'category': HallucinationType.CONTRADICTORY,
                'weight': self.HALLUCINATION_WEIGHTS[HallucinationType.CONTRADICTORY],
                'reason': 'Claim directly contradicts ground truth'
            }
        
        # Check for fabricated citations in the claim
        claim_start = response.find(claim)
        if claim_start >= 0:
            claim_end = claim_start + len(claim)
            for cit_pos in citation_positions:
                if claim_start <= cit_pos[0] < claim_end or claim_start <= cit_pos[1] < claim_end:
                    return {
                        'claim': claim,
                        'category': HallucinationType.FABRICATED_CITATION,
                        'weight': self.HALLUCINATION_WEIGHTS[HallucinationType.FABRICATED_CITATION],
                        'reason': 'Claim contains fabricated citation'
                    }
        
        # Default to unsupported claim
        return {
            'claim': claim,
            'category': HallucinationType.UNSUPPORTED_CLAIM,
            'weight': self.HALLUCINATION_WEIGHTS[HallucinationType.UNSUPPORTED_CLAIM],
            'reason': 'Claim is not supported by ground truth (but not contradictory)'
        }
    
    def _is_opposite_claim(self, claim: str, ground_claim: str) -> bool:
        """Check if a claim is the opposite of a ground truth claim."""
        # Common antonym pairs
        antonym_pairs = [
            ('increases', 'decreases'), ('increase', 'decrease'),
            ('higher', 'lower'), ('high', 'low'),
            ('more', 'less'), ('larger', 'smaller'),
            ('before', 'after'), ('above', 'below'),
            ('positive', 'negative'), ('true', 'false'),
            ('yes', 'no'), ('is', 'is not'), ('was', 'was not')
        ]
        
        for pos, neg in antonym_pairs:
            if (pos in claim and neg in ground_claim) or (neg in claim and pos in ground_claim):
                # Check if the rest of the claim is similar
                claim_clean = claim.replace(pos, '').replace(neg, '').strip()
                ground_clean = ground_claim.replace(pos, '').replace(neg, '').strip()
                if self._has_significant_overlap(claim_clean, ground_clean):
                    return True
        
        return False
    
    def _has_significant_overlap(self, text1: str, text2: str) -> bool:
        """Check if two texts have significant word overlap."""
        words1 = set(self._tokenize(text1))
        words2 = set(self._tokenize(text2))
        
        if not words1 or not words2:
            return False
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return (intersection / union) > 0.3 if union > 0 else False
    
    def _calculate_weighted_hallucination_rate(self, classifications: List[Dict[str, Any]]) -> float:
        """
        Calculate weighted hallucination rate based on claim classifications.
        
        Uses severity weights to give more importance to serious hallucinations
        like contradictions and fabricated citations.
        """
        if not classifications:
            return 0.0
        
        total_weight = 0.0
        max_possible_weight = 0.0
        
        for classification in classifications:
            weight = classification['weight']
            category = classification['category']
            
            # Accumulate weighted score
            total_weight += weight
            max_possible_weight += self.HALLUCINATION_WEIGHTS[HallucinationType.CONTRADICTORY]  # Max weight
        
        # Normalize to 0-1 scale
        if max_possible_weight == 0:
            return 0.0
        
        weighted_rate = total_weight / max_possible_weight
        return round(min(weighted_rate, 1.0), 4)
    
    def calculate_hallucination_rate(self, claims: List[str], ground_truth: str) -> float:
        """
        DEPRECATED: Simple hallucination rate as inverse of consistency.
        Use classify_claims() for granular classification.
        """
        consistency = self.calculate_factual_consistency(claims, ground_truth)
        return round(1.0 - consistency, 4)
    
    def detect_fabricated_citations(self, response: str) -> List[Dict[str, Any]]:
        """
        Detect potentially fabricated citations in the response.
        Looks for citation patterns that don't match known sources.
        """
        fabricated = []
        
        # Pattern for common citation formats
        citation_patterns = [
            r'\[\d+\]',  # [1], [2], etc.
            r'\(\w+\s+et\s+al\.?,?\s*\d{4}[a-z]?\)',  # (Smith et al., 2020)
            r'\(\w+,?\s*\d{4}[a-z]?\)',  # (Smith, 2020)
            r'\[\w+\s+et\s+al\.?,?\s*\d{4}\]',  # [Smith et al., 2020]
            r'\d{4}[a-z]?\s+et\s+al\.',  # 2020 et al.
        ]
        
        for pattern in citation_patterns:
            matches = re.finditer(pattern, response)
            for match in matches:
                fabricated.append({
                    'citation': match.group(),
                    'position': match.span(),
                    'pattern': pattern,
                    'verified': False  # Would need external verification
                })
        
        return fabricated
    
    def extract_claims(self, text: str) -> List[str]:
        """
        Extract factual claims from text.
        Simple implementation using sentence splitting.
        """
        # Split into sentences
        sentences = re.split(r'[.!?]+', text)
        claims = []
        
        for sent in sentences:
            sent = sent.strip()
            if len(sent) > 10:  # Filter out very short fragments
                # Filter out questions and subjective statements
                if not sent.endswith('?') and not sent.lower().startswith(('i think', 'i believe', 'maybe', 'perhaps')):
                    claims.append(sent)
        
        return claims
    
    def calculate_bleu_score(self, response: str, ground_truth: str, n: int = 4) -> float:
        """
        Calculate BLEU score for response against ground truth.
        """
        response_tokens = self._tokenize(response.lower())
        ground_tokens = self._tokenize(ground_truth.lower())
        
        if not ground_tokens or not response_tokens:
            return 0.0
        
        # Calculate n-gram precisions
        precisions = []
        for i in range(1, n + 1):
            precision = self._ngram_precision(response_tokens, ground_tokens, i)
            precisions.append(precision)
        
        # Geometric mean with brevity penalty
        if min(precisions) > 0:
            geo_mean = np.exp(np.mean(np.log(precisions)))
        else:
            geo_mean = 0.0
        
        # Brevity penalty
        bp = min(1.0, np.exp(1 - len(ground_tokens) / len(response_tokens)))
        
        return round(geo_mean * bp, 4)
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        # Remove punctuation and split
        text = text.translate(str.maketrans('', '', string.punctuation))
        return text.split()
    
    def _ngram_precision(self, candidate: List[str], reference: List[str], n: int) -> float:
        """Calculate n-gram precision."""
        candidate_ngrams = self._get_ngrams(candidate, n)
        reference_ngrams = self._get_ngrams(reference, n)
        
        if not candidate_ngrams:
            return 0.0
        
        matches = sum((Counter(candidate_ngrams) & Counter(reference_ngrams)).values())
        return matches / len(candidate_ngrams)
    
    def _get_ngrams(self, tokens: List[str], n: int) -> List[Tuple]:
        """Generate n-grams from tokens."""
        return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
    
    def _claim_supported(self, claim: str, ground_truth: str) -> bool:
        """Check if a claim is supported by ground truth."""
        # Exact substring match
        if claim in ground_truth:
            return True
        
        # Check for significant word overlap (Jaccard similarity)
        claim_words = set(self._tokenize(claim))
        ground_words = set(self._tokenize(ground_truth))
        
        if not claim_words:
            return False
        
        intersection = len(claim_words & ground_words)
        union = len(claim_words | ground_words)
        
        jaccard = intersection / union if union > 0 else 0
        return jaccard > 0.5  # Threshold for support
    
    def compute_all_metrics(self, response: str, ground_truth: str) -> Dict[str, Any]:
        """Compute all metrics for a single response."""
        claims = self.extract_claims(response)
        citations = self.detect_fabricated_citations(response)
        
        # Get granular classification
        classification_result = self.classify_claims(claims, response, ground_truth)
        
        # Use weighted hallucination rate instead of simple inverse
        weighted_hallucination_rate = classification_result['weighted_hallucination_rate']
        
        metrics = {
            'faithfulness': self.calculate_faithfulness(response, ground_truth),
            'factual_consistency': self.calculate_factual_consistency(claims, ground_truth),
            'hallucination_rate': weighted_hallucination_rate,  # Now uses weighted calculation
            'simple_hallucination_rate': self.calculate_hallucination_rate(claims, ground_truth),  # Keep for reference
            'bleu_score': self.calculate_bleu_score(response, ground_truth),
            'claim_count': len(claims),
            'citation_count': len(citations),
            'fabricated_citations': citations,
            'claims': claims,
            'claim_classifications': classification_result['classifications'],
            'hallucination_category_counts': classification_result['category_counts']
        }
        
        return metrics
