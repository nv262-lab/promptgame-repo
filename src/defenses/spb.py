"""
Semantic Prompt Binding (SPB) Defense Mechanism

Algorithm 1 from the paper:
Enforces semantic consistency between defended outputs and reference outputs.
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass
import torch
from transformers import AutoTokenizer, AutoModel
from bert_score import score as bert_score
import hashlib


@dataclass
class SPBResult:
    """Result of SPB defense application"""
    original_input: str
    defended_output: str
    reference_output: str
    semantic_similarity: float
    is_attack_blocked: bool
    defense_action: str  # "accept", "reject", "modify"
    processing_time_ms: float


class SemanticPromptBinding:
    """
    Semantic Prompt Binding (SPB) Defense
    
    Algorithm 1: Semantic Prompt Binding
    Input: System prompt s, user input x, LLM M, threshold τ
    Output: Defended output y or rejection
    
    1: y_ref ← M(s, x_benign)  // Compute reference output
    2: y ← M(s, x)  // Compute actual output
    3: σ ← SemanticSimilarity(y, y_ref)
    4: if σ ≥ τ then
    5:     return y  // Accept output
    6: else
    7:     return y_ref  // Reject and return reference
    8: end if
    """
    
    def __init__(
        self,
        threshold: float = 0.85,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        use_bert_score: bool = True,
        cache_references: bool = True
    ):
        """
        Initialize SPB defense.
        
        Args:
            threshold: Semantic similarity threshold τ (default 0.85)
            embedding_model: Model for computing embeddings
            use_bert_score: Use BERTScore for similarity (recommended)
            cache_references: Cache reference outputs for efficiency
        """
        self.threshold = threshold
        self.use_bert_score = use_bert_score
        self.cache_references = cache_references
        self.reference_cache: Dict[str, str] = {}
        
        # Load embedding model for fallback similarity
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if not use_bert_score:
            self.tokenizer = AutoTokenizer.from_pretrained(embedding_model)
            self.model = AutoModel.from_pretrained(embedding_model).to(self.device)
            self.model.eval()
    
    def _cache_key(self, system_prompt: str, benign_input: str) -> str:
        """Generate cache key for reference outputs"""
        content = f"{system_prompt}||{benign_input}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def compute_reference_output(
        self,
        system_prompt: str,
        model_func,
        benign_input: str = "Hello, how can I help you today?"
    ) -> str:
        """
        Compute reference output y_ref = M(s, x_benign)
        
        Args:
            system_prompt: The system prompt s
            model_func: LLM function M
            benign_input: Benign input x_benign
        
        Returns:
            Reference output y_ref
        """
        if self.cache_references:
            cache_key = self._cache_key(system_prompt, benign_input)
            if cache_key in self.reference_cache:
                return self.reference_cache[cache_key]
        
        reference_output = model_func(system_prompt, benign_input)
        
        if self.cache_references:
            self.reference_cache[cache_key] = reference_output
        
        return reference_output
    
    def compute_semantic_similarity(
        self,
        output: str,
        reference: str
    ) -> float:
        """
        Compute semantic similarity σ(y, y_ref)
        
        Uses BERTScore F1 as the semantic similarity metric.
        
        Args:
            output: Model output y
            reference: Reference output y_ref
        
        Returns:
            Semantic similarity score in [0, 1]
        """
        if self.use_bert_score:
            # Use BERTScore F1
            P, R, F1 = bert_score(
                [output], [reference],
                lang="en",
                verbose=False,
                device=self.device
            )
            return F1.item()
        else:
            # Fallback: cosine similarity of embeddings
            return self._embedding_similarity(output, reference)
    
    def _embedding_similarity(self, text1: str, text2: str) -> float:
        """Compute cosine similarity using embeddings"""
        with torch.no_grad():
            # Tokenize
            inputs1 = self.tokenizer(
                text1, return_tensors="pt",
                truncation=True, max_length=512, padding=True
            ).to(self.device)
            inputs2 = self.tokenizer(
                text2, return_tensors="pt",
                truncation=True, max_length=512, padding=True
            ).to(self.device)
            
            # Get embeddings (mean pooling)
            outputs1 = self.model(**inputs1)
            outputs2 = self.model(**inputs2)
            
            emb1 = outputs1.last_hidden_state.mean(dim=1)
            emb2 = outputs2.last_hidden_state.mean(dim=1)
            
            # Cosine similarity
            similarity = torch.nn.functional.cosine_similarity(emb1, emb2)
            return similarity.item()
    
    def defend(
        self,
        system_prompt: str,
        user_input: str,
        model_func,
        benign_input: str = "Hello, how can I help you today?"
    ) -> SPBResult:
        """
        Apply SPB defense (Algorithm 1).
        
        Args:
            system_prompt: System prompt s
            user_input: User input x (potentially adversarial)
            model_func: LLM function M
            benign_input: Benign reference input
        
        Returns:
            SPBResult with defense outcome
        """
        import time
        start_time = time.time()
        
        # Line 1: Compute reference output
        reference_output = self.compute_reference_output(
            system_prompt, model_func, benign_input
        )
        
        # Line 2: Compute actual output
        actual_output = model_func(system_prompt, user_input)
        
        # Line 3: Compute semantic similarity
        similarity = self.compute_semantic_similarity(actual_output, reference_output)
        
        # Lines 4-8: Decision
        if similarity >= self.threshold:
            # Accept output
            defended_output = actual_output
            is_blocked = False
            action = "accept"
        else:
            # Reject and return reference
            defended_output = reference_output
            is_blocked = True
            action = "reject"
        
        processing_time = (time.time() - start_time) * 1000  # Convert to ms
        
        return SPBResult(
            original_input=user_input,
            defended_output=defended_output,
            reference_output=reference_output,
            semantic_similarity=similarity,
            is_attack_blocked=is_blocked,
            defense_action=action,
            processing_time_ms=processing_time
        )
    
    def batch_defend(
        self,
        system_prompt: str,
        inputs: list,
        model_func,
        benign_input: str = "Hello, how can I help you today?"
    ) -> list:
        """
        Apply SPB defense to multiple inputs.
        
        Args:
            system_prompt: System prompt s
            inputs: List of user inputs
            model_func: LLM function M
            benign_input: Benign reference input
        
        Returns:
            List of SPBResult objects
        """
        results = []
        for user_input in inputs:
            result = self.defend(system_prompt, user_input, model_func, benign_input)
            results.append(result)
        return results
    
    def clear_cache(self):
        """Clear reference output cache"""
        self.reference_cache.clear()


def main():
    """Test SPB defense"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test SPB Defense")
    parser.add_argument("--input", type=str, required=True, help="User input to test")
    parser.add_argument("--system-prompt", type=str, default="You are a helpful assistant.", help="System prompt")
    parser.add_argument("--threshold", type=float, default=0.85, help="Semantic threshold")
    
    args = parser.parse_args()
    
    # Mock model function for testing
    def mock_model(system_prompt: str, user_input: str) -> str:
        return f"Response to: {user_input[:50]}..."
    
    spb = SemanticPromptBinding(threshold=args.threshold)
    result = spb.defend(args.system_prompt, args.input, mock_model)
    
    print(f"Input: {result.original_input}")
    print(f"Similarity: {result.semantic_similarity:.4f}")
    print(f"Action: {result.defense_action}")
    print(f"Blocked: {result.is_attack_blocked}")
    print(f"Processing time: {result.processing_time_ms:.2f}ms")


if __name__ == "__main__":
    main()
