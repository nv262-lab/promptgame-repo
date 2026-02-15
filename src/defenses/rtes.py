"""
Randomized Token Embedding Shuffling (RTES) Defense Mechanism

Algorithm 3 from the paper:
Implements randomized defense strategy to prevent fingerprinting attacks,
achieving Commitment-Resistant Equilibrium (CRE).
"""

import numpy as np
import secrets
import hashlib
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import time


class ShuffleStrategy(Enum):
    """Token shuffling strategies"""
    UNIFORM = "uniform"
    WEIGHTED = "weighted"
    ADAPTIVE = "adaptive"


@dataclass
class RTESConfig:
    """Configuration for RTES defense"""
    entropy_bits: int = 8
    shuffle_strategy: ShuffleStrategy = ShuffleStrategy.ADAPTIVE
    min_tokens_to_shuffle: int = 3
    max_shuffle_fraction: float = 0.3
    preserve_semantics: bool = True
    seed: Optional[int] = None


@dataclass
class RTESResult:
    """Result of RTES defense application"""
    original_input: str
    shuffled_input: str
    tokens_shuffled: int
    total_tokens: int
    shuffle_fraction: float
    entropy_used: float
    strategy_selected: str
    processing_time_ms: float


class RandomizedTokenEmbeddingShuffling:
    """
    Randomized Token Embedding Shuffling (RTES) Defense
    
    Algorithm 3: Randomized Token Embedding Shuffling
    Input: User input x, entropy bits b, strategy set Σ
    Output: Shuffled input x', strategy metadata
    
    1: r ← SecureRandom(b)  // Generate b bits of randomness
    2: σ ← SelectStrategy(Σ, r)  // Select shuffling strategy
    3: tokens ← Tokenize(x)
    4: shuffle_set ← SelectTokens(tokens, σ, r)
    5: for each token t in shuffle_set do
    6:     t' ← ShuffleEmbedding(t, r)
    7:     Replace t with t' in tokens
    8: end for
    9: x' ← Detokenize(tokens)
    10: return x'
    
    This implements the randomness requirement from Theorem 3:
    Any defense maintaining SRE requires Ω(log n) bits of randomness.
    """
    
    # Semantically equivalent token groups for shuffling
    SEMANTIC_GROUPS = {
        "articles": ["a", "an", "the"],
        "conjunctions": ["and", "but", "or", "yet", "so"],
        "prepositions": ["in", "on", "at", "by", "to", "for", "with"],
        "pronouns": ["i", "you", "he", "she", "it", "we", "they"],
        "quantifiers": ["some", "any", "many", "few", "several"],
        "auxiliaries": ["is", "are", "was", "were", "be", "been"],
        "modals": ["can", "could", "may", "might", "will", "would", "should"],
    }
    
    # Character substitution map for visual similarity
    CHAR_SUBSTITUTIONS = {
        'a': ['а', 'ɑ', 'α'],  # Cyrillic a, Latin alpha
        'e': ['е', 'ε', 'ė'],  # Cyrillic e, Greek epsilon
        'o': ['о', 'ο', 'ø'],  # Cyrillic o, Greek omicron
        'i': ['і', 'ι', 'ı'],  # Cyrillic i, Greek iota
        'c': ['с', 'ϲ'],  # Cyrillic s
        'p': ['р', 'ρ'],  # Cyrillic r, Greek rho
        's': ['ѕ', 'ꜱ'],  # Cyrillic dze
    }
    
    def __init__(self, config: Optional[RTESConfig] = None):
        """
        Initialize RTES defense.
        
        Args:
            config: RTES configuration (uses defaults if None)
        """
        self.config = config or RTESConfig()
        self.rng = np.random.default_rng(self.config.seed)
        
        # Build reverse lookup for semantic groups
        self.token_to_group = {}
        for group_name, tokens in self.SEMANTIC_GROUPS.items():
            for token in tokens:
                self.token_to_group[token.lower()] = group_name
        
        # Track strategy distribution for CRE analysis
        self.strategy_history: List[str] = []
    
    def _generate_randomness(self) -> bytes:
        """
        Generate cryptographically secure random bits (Line 1).
        
        Returns:
            Random bytes with specified entropy
        """
        num_bytes = (self.config.entropy_bits + 7) // 8
        return secrets.token_bytes(num_bytes)
    
    def _select_strategy(self, random_bits: bytes) -> ShuffleStrategy:
        """
        Select shuffling strategy based on randomness (Line 2).
        
        Ensures unpredictability for CRE compliance.
        
        Args:
            random_bits: Random bytes for selection
        
        Returns:
            Selected ShuffleStrategy
        """
        if self.config.shuffle_strategy != ShuffleStrategy.ADAPTIVE:
            return self.config.shuffle_strategy
        
        # Use first byte to select strategy
        selector = random_bits[0] % 3
        strategies = [ShuffleStrategy.UNIFORM, ShuffleStrategy.WEIGHTED, ShuffleStrategy.ADAPTIVE]
        return strategies[selector]
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Simple whitespace tokenization (Line 3).
        
        For production, use proper tokenizer matching the target LLM.
        """
        return text.split()
    
    def _detokenize(self, tokens: List[str]) -> str:
        """Reconstruct text from tokens (Line 9)."""
        return " ".join(tokens)
    
    def _select_tokens_to_shuffle(
        self,
        tokens: List[str],
        strategy: ShuffleStrategy,
        random_bits: bytes
    ) -> List[int]:
        """
        Select which tokens to shuffle (Line 4).
        
        Args:
            tokens: List of tokens
            strategy: Shuffling strategy
            random_bits: Random bytes for selection
        
        Returns:
            List of token indices to shuffle
        """
        n_tokens = len(tokens)
        if n_tokens < self.config.min_tokens_to_shuffle:
            return []
        
        max_to_shuffle = int(n_tokens * self.config.max_shuffle_fraction)
        max_to_shuffle = max(max_to_shuffle, self.config.min_tokens_to_shuffle)
        
        # Determine number to shuffle based on strategy
        if strategy == ShuffleStrategy.UNIFORM:
            # Uniform random selection
            n_shuffle = secrets.randbelow(max_to_shuffle) + 1
            indices = list(range(n_tokens))
            secrets.SystemRandom().shuffle(indices)
            return sorted(indices[:n_shuffle])
        
        elif strategy == ShuffleStrategy.WEIGHTED:
            # Prefer shuffling certain positions (beginning, end)
            weights = np.array([
                1.5 if i < 3 or i >= n_tokens - 3 else 1.0
                for i in range(n_tokens)
            ])
            weights /= weights.sum()
            n_shuffle = min(max_to_shuffle, max(1, int(random_bits[1] % max_to_shuffle) + 1))
            indices = self.rng.choice(n_tokens, size=n_shuffle, replace=False, p=weights)
            return sorted(indices.tolist())
        
        else:  # ADAPTIVE
            # Select based on token type (prefer function words)
            candidate_indices = []
            for i, token in enumerate(tokens):
                if token.lower() in self.token_to_group:
                    candidate_indices.append(i)
            
            if not candidate_indices:
                # Fall back to uniform if no candidates
                return self._select_tokens_to_shuffle(tokens, ShuffleStrategy.UNIFORM, random_bits)
            
            n_shuffle = min(len(candidate_indices), max_to_shuffle)
            n_shuffle = max(1, int(random_bits[1] % n_shuffle) + 1)
            selected = self.rng.choice(candidate_indices, size=min(n_shuffle, len(candidate_indices)), replace=False)
            return sorted(selected.tolist())
    
    def _shuffle_token(self, token: str, random_bits: bytes, index: int) -> str:
        """
        Apply shuffling transformation to a token (Lines 5-8).
        
        Args:
            token: Original token
            random_bits: Random bytes
            index: Token index for deterministic variation
        
        Returns:
            Shuffled token
        """
        if self.config.preserve_semantics:
            return self._semantic_shuffle(token, random_bits, index)
        else:
            return self._character_shuffle(token, random_bits, index)
    
    def _semantic_shuffle(self, token: str, random_bits: bytes, index: int) -> str:
        """
        Shuffle token while preserving semantic meaning.
        
        Replaces token with semantically equivalent alternative.
        """
        token_lower = token.lower()
        
        if token_lower in self.token_to_group:
            group_name = self.token_to_group[token_lower]
            alternatives = self.SEMANTIC_GROUPS[group_name]
            
            # Select alternative based on randomness
            selector = (random_bits[index % len(random_bits)] + index) % len(alternatives)
            replacement = alternatives[selector]
            
            # Preserve case
            if token[0].isupper():
                replacement = replacement.capitalize()
            if token.isupper():
                replacement = replacement.upper()
            
            return replacement
        
        return token
    
    def _character_shuffle(self, token: str, random_bits: bytes, index: int) -> str:
        """
        Shuffle token using character substitutions.
        
        Uses visually similar characters to modify tokens.
        """
        result = list(token)
        
        for i, char in enumerate(result):
            char_lower = char.lower()
            if char_lower in self.CHAR_SUBSTITUTIONS:
                # Probabilistic substitution
                if (random_bits[(index + i) % len(random_bits)] % 4) == 0:
                    alternatives = self.CHAR_SUBSTITUTIONS[char_lower]
                    selector = random_bits[(index + i + 1) % len(random_bits)] % len(alternatives)
                    result[i] = alternatives[selector]
        
        return "".join(result)
    
    def apply(self, user_input: str) -> RTESResult:
        """
        Apply RTES defense (Algorithm 3).
        
        Args:
            user_input: User input x
        
        Returns:
            RTESResult with shuffled input and metadata
        """
        start_time = time.time()
        
        # Line 1: Generate randomness
        random_bits = self._generate_randomness()
        
        # Line 2: Select strategy
        strategy = self._select_strategy(random_bits)
        self.strategy_history.append(strategy.value)
        
        # Line 3: Tokenize
        tokens = self._tokenize(user_input)
        
        if len(tokens) == 0:
            return RTESResult(
                original_input=user_input,
                shuffled_input=user_input,
                tokens_shuffled=0,
                total_tokens=0,
                shuffle_fraction=0.0,
                entropy_used=0.0,
                strategy_selected=strategy.value,
                processing_time_ms=(time.time() - start_time) * 1000
            )
        
        # Line 4: Select tokens to shuffle
        shuffle_indices = self._select_tokens_to_shuffle(tokens, strategy, random_bits)
        
        # Lines 5-8: Apply shuffling
        shuffled_tokens = tokens.copy()
        for idx in shuffle_indices:
            shuffled_tokens[idx] = self._shuffle_token(tokens[idx], random_bits, idx)
        
        # Line 9: Detokenize
        shuffled_input = self._detokenize(shuffled_tokens)
        
        # Compute entropy used
        entropy_used = self.config.entropy_bits * (len(shuffle_indices) / max(len(tokens), 1))
        
        processing_time = (time.time() - start_time) * 1000
        
        return RTESResult(
            original_input=user_input,
            shuffled_input=shuffled_input,
            tokens_shuffled=len(shuffle_indices),
            total_tokens=len(tokens),
            shuffle_fraction=len(shuffle_indices) / max(len(tokens), 1),
            entropy_used=entropy_used,
            strategy_selected=strategy.value,
            processing_time_ms=processing_time
        )
    
    def compute_strategy_entropy(self) -> float:
        """
        Compute entropy of strategy selection history.
        
        Used for Table VIII (Equilibrium Convergence Analysis).
        High entropy indicates unpredictable defense (good for CRE).
        
        Returns:
            Shannon entropy of strategy distribution
        """
        if not self.strategy_history:
            return 0.0
        
        # Count strategy frequencies
        counts = {}
        for strategy in self.strategy_history:
            counts[strategy] = counts.get(strategy, 0) + 1
        
        # Compute entropy
        total = len(self.strategy_history)
        entropy = 0.0
        for count in counts.values():
            prob = count / total
            if prob > 0:
                entropy -= prob * np.log2(prob)
        
        return entropy
    
    def compute_fingerprint_resistance(self, n_samples: int = 100) -> Dict[str, float]:
        """
        Compute fingerprinting resistance metrics.
        
        Used for Table IX (Defense Fingerprinting Resistance).
        
        Args:
            n_samples: Number of samples for estimation
        
        Returns:
            Dictionary with resistance metrics
        """
        test_input = "This is a test input with some words that can be shuffled."
        
        outputs = set()
        strategy_counts = {}
        
        for _ in range(n_samples):
            result = self.apply(test_input)
            outputs.add(result.shuffled_input)
            strategy_counts[result.strategy_selected] = strategy_counts.get(result.strategy_selected, 0) + 1
        
        # Unique output ratio (higher = more resistant)
        unique_ratio = len(outputs) / n_samples
        
        # Strategy distribution entropy
        total = sum(strategy_counts.values())
        strategy_entropy = 0.0
        for count in strategy_counts.values():
            prob = count / total
            if prob > 0:
                strategy_entropy -= prob * np.log2(prob)
        
        # Maximum strategy probability (lower = more resistant)
        max_strategy_prob = max(strategy_counts.values()) / total
        
        return {
            "unique_output_ratio": unique_ratio,
            "strategy_entropy": strategy_entropy,
            "max_strategy_probability": max_strategy_prob,
            "effective_randomness_bits": np.log2(len(outputs)) if len(outputs) > 1 else 0
        }
    
    def reset_history(self):
        """Reset strategy history"""
        self.strategy_history.clear()


def main():
    """Test RTES defense"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test RTES Defense")
    parser.add_argument("--input", type=str, required=True, help="User input to shuffle")
    parser.add_argument("--entropy-bits", type=int, default=8, help="Entropy bits")
    parser.add_argument("--n-samples", type=int, default=10, help="Number of samples to show")
    
    args = parser.parse_args()
    
    config = RTESConfig(entropy_bits=args.entropy_bits)
    rtes = RandomizedTokenEmbeddingShuffling(config)
    
    print(f"Original: {args.input}")
    print(f"\nShuffled samples ({args.n_samples}):")
    
    for i in range(args.n_samples):
        result = rtes.apply(args.input)
        print(f"  {i+1}. {result.shuffled_input}")
        print(f"     Strategy: {result.strategy_selected}, Tokens shuffled: {result.tokens_shuffled}/{result.total_tokens}")
    
    # Compute fingerprint resistance
    print(f"\nFingerprint Resistance Metrics:")
    metrics = rtes.compute_fingerprint_resistance(100)
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")


if __name__ == "__main__":
    main()
