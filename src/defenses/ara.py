"""
Adaptive Risk Assessment (ARA) Defense Mechanism

Algorithm 2 from the paper:
Dynamically adjusts defense intensity based on input risk scoring.
"""

import re
import numpy as np
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
from enum import Enum
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class RiskLevel(Enum):
    """Risk levels for ARA assessment"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class DefenseAction(Enum):
    """Defense actions based on risk level"""
    ALLOW = "allow"
    SANITIZE = "sanitize"
    QUARANTINE = "quarantine"
    REJECT = "reject"


@dataclass
class ARARiskFeatures:
    """Features extracted for risk assessment"""
    instruction_keywords: int
    separator_patterns: int
    encoding_anomalies: int
    length_ratio: float
    entropy_score: float
    repetition_score: float
    special_char_ratio: float


@dataclass
class ARAResult:
    """Result of ARA risk assessment"""
    original_input: str
    risk_score: float
    risk_level: RiskLevel
    defense_action: DefenseAction
    sanitized_input: Optional[str]
    features: ARARiskFeatures
    processing_time_ms: float


class AdaptiveRiskAssessment:
    """
    Adaptive Risk Assessment (ARA) Defense
    
    Algorithm 2: Adaptive Risk Assessment
    Input: User input x, risk thresholds T = [t_1, t_2, t_3]
    Output: Risk level r, defense action a
    
    1: f ← ExtractFeatures(x)  // Extract risk features
    2: r_score ← RiskModel(f)  // Compute risk score
    3: if r_score < t_1 then
    4:     return (LOW, ALLOW)
    5: else if r_score < t_2 then
    6:     return (MEDIUM, SANITIZE)
    7: else if r_score < t_3 then
    8:     return (HIGH, QUARANTINE)
    9: else
    10:    return (CRITICAL, REJECT)
    11: end if
    """
    
    # Known injection patterns
    INSTRUCTION_KEYWORDS = [
        "ignore", "disregard", "forget", "override", "bypass",
        "instead", "actually", "new instruction", "system prompt",
        "you are now", "pretend", "roleplay", "act as",
        "ignore previous", "ignore above", "ignore all",
        "do not follow", "stop being", "jailbreak",
        "developer mode", "admin mode", "sudo", "root access"
    ]
    
    SEPARATOR_PATTERNS = [
        r"---+", r"===+", r"\*\*\*+", r"```", r"~~~",
        r"</?system>", r"</?user>", r"</?assistant>",
        r"\[INST\]", r"\[/INST\]", r"<<SYS>>", r"<</SYS>>",
        r"###\s*(instruction|system|user)", r"human:", r"assistant:",
        r"<\|.*?\|>", r"\[START\]", r"\[END\]"
    ]
    
    ENCODING_PATTERNS = [
        r"\\x[0-9a-fA-F]{2}", r"\\u[0-9a-fA-F]{4}",
        r"%[0-9a-fA-F]{2}", r"&#\d+;", r"&#x[0-9a-fA-F]+;",
        r"base64:", r"data:text"
    ]
    
    def __init__(
        self,
        risk_thresholds: List[float] = [0.3, 0.6, 0.9],
        use_ml_model: bool = False,
        ml_model_path: Optional[str] = None,
        feature_weights: Optional[Dict[str, float]] = None
    ):
        """
        Initialize ARA defense.
        
        Args:
            risk_thresholds: [t_1, t_2, t_3] thresholds for risk levels
            use_ml_model: Use ML-based risk scoring
            ml_model_path: Path to trained risk model
            feature_weights: Weights for heuristic features
        """
        assert len(risk_thresholds) == 3, "Need exactly 3 thresholds"
        assert all(risk_thresholds[i] < risk_thresholds[i+1] for i in range(2)), \
            "Thresholds must be increasing"
        
        self.thresholds = risk_thresholds
        self.use_ml_model = use_ml_model
        
        # Default feature weights (tuned on validation set)
        self.feature_weights = feature_weights or {
            "instruction_keywords": 0.25,
            "separator_patterns": 0.20,
            "encoding_anomalies": 0.15,
            "length_ratio": 0.10,
            "entropy_score": 0.10,
            "repetition_score": 0.10,
            "special_char_ratio": 0.10
        }
        
        # Compile regex patterns
        self.separator_regex = [re.compile(p, re.IGNORECASE) for p in self.SEPARATOR_PATTERNS]
        self.encoding_regex = [re.compile(p, re.IGNORECASE) for p in self.ENCODING_PATTERNS]
        
        # Load ML model if specified
        if use_ml_model and ml_model_path:
            self._load_ml_model(ml_model_path)
    
    def _load_ml_model(self, model_path: str):
        """Load trained ML risk model"""
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.ml_model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.ml_model.eval()
    
    def _count_instruction_keywords(self, text: str) -> int:
        """Count instruction-like keywords"""
        text_lower = text.lower()
        count = 0
        for keyword in self.INSTRUCTION_KEYWORDS:
            count += text_lower.count(keyword)
        return count
    
    def _count_separator_patterns(self, text: str) -> int:
        """Count separator/delimiter patterns"""
        count = 0
        for pattern in self.separator_regex:
            count += len(pattern.findall(text))
        return count
    
    def _count_encoding_anomalies(self, text: str) -> int:
        """Count encoding anomalies (hex, unicode escapes, etc.)"""
        count = 0
        for pattern in self.encoding_regex:
            count += len(pattern.findall(text))
        return count
    
    def _compute_length_ratio(self, text: str, max_expected: int = 500) -> float:
        """Compute normalized length ratio"""
        return min(len(text) / max_expected, 2.0) / 2.0
    
    def _compute_entropy(self, text: str) -> float:
        """Compute character-level entropy"""
        if not text:
            return 0.0
        
        char_counts = {}
        for char in text:
            char_counts[char] = char_counts.get(char, 0) + 1
        
        length = len(text)
        entropy = 0.0
        for count in char_counts.values():
            prob = count / length
            entropy -= prob * np.log2(prob)
        
        # Normalize to [0, 1] (max entropy for ASCII is ~7 bits)
        return min(entropy / 7.0, 1.0)
    
    def _compute_repetition_score(self, text: str) -> float:
        """Compute repetition score (repeated n-grams)"""
        if len(text) < 10:
            return 0.0
        
        # Check for repeated 3-grams
        ngram_size = 3
        ngrams = [text[i:i+ngram_size] for i in range(len(text) - ngram_size + 1)]
        unique_ngrams = set(ngrams)
        
        if len(ngrams) == 0:
            return 0.0
        
        repetition_ratio = 1.0 - (len(unique_ngrams) / len(ngrams))
        return repetition_ratio
    
    def _compute_special_char_ratio(self, text: str) -> float:
        """Compute ratio of special characters"""
        if not text:
            return 0.0
        
        special_chars = sum(1 for c in text if not c.isalnum() and not c.isspace())
        return special_chars / len(text)
    
    def extract_features(self, text: str) -> ARARiskFeatures:
        """
        Extract risk features from input text (Line 1 of Algorithm 2).
        
        Args:
            text: User input x
        
        Returns:
            ARARiskFeatures with all extracted features
        """
        return ARARiskFeatures(
            instruction_keywords=self._count_instruction_keywords(text),
            separator_patterns=self._count_separator_patterns(text),
            encoding_anomalies=self._count_encoding_anomalies(text),
            length_ratio=self._compute_length_ratio(text),
            entropy_score=self._compute_entropy(text),
            repetition_score=self._compute_repetition_score(text),
            special_char_ratio=self._compute_special_char_ratio(text)
        )
    
    def compute_risk_score(self, features: ARARiskFeatures) -> float:
        """
        Compute risk score from features (Line 2 of Algorithm 2).
        
        Args:
            features: Extracted risk features
        
        Returns:
            Risk score in [0, 1]
        """
        if self.use_ml_model:
            return self._ml_risk_score(features)
        
        # Heuristic scoring
        score = 0.0
        
        # Instruction keywords (normalize by max expected)
        keyword_score = min(features.instruction_keywords / 5.0, 1.0)
        score += self.feature_weights["instruction_keywords"] * keyword_score
        
        # Separator patterns
        separator_score = min(features.separator_patterns / 3.0, 1.0)
        score += self.feature_weights["separator_patterns"] * separator_score
        
        # Encoding anomalies
        encoding_score = min(features.encoding_anomalies / 5.0, 1.0)
        score += self.feature_weights["encoding_anomalies"] * encoding_score
        
        # Length ratio (longer inputs are riskier)
        score += self.feature_weights["length_ratio"] * features.length_ratio
        
        # Entropy (very low or very high entropy is suspicious)
        entropy_risk = abs(features.entropy_score - 0.5) * 2  # Distance from normal
        score += self.feature_weights["entropy_score"] * entropy_risk
        
        # Repetition (high repetition is suspicious)
        score += self.feature_weights["repetition_score"] * features.repetition_score
        
        # Special characters
        special_risk = min(features.special_char_ratio * 2, 1.0)
        score += self.feature_weights["special_char_ratio"] * special_risk
        
        return min(score, 1.0)
    
    def _ml_risk_score(self, features: ARARiskFeatures) -> float:
        """Compute risk score using ML model"""
        # This would use a trained classifier
        # For now, fall back to heuristic
        return self.compute_risk_score(features)
    
    def determine_risk_level(self, risk_score: float) -> RiskLevel:
        """
        Determine risk level from score (Lines 3-11 of Algorithm 2).
        
        Args:
            risk_score: Computed risk score
        
        Returns:
            RiskLevel enum value
        """
        if risk_score < self.thresholds[0]:
            return RiskLevel.LOW
        elif risk_score < self.thresholds[1]:
            return RiskLevel.MEDIUM
        elif risk_score < self.thresholds[2]:
            return RiskLevel.HIGH
        else:
            return RiskLevel.CRITICAL
    
    def determine_action(self, risk_level: RiskLevel) -> DefenseAction:
        """
        Determine defense action based on risk level.
        
        Args:
            risk_level: Assessed risk level
        
        Returns:
            DefenseAction enum value
        """
        action_map = {
            RiskLevel.LOW: DefenseAction.ALLOW,
            RiskLevel.MEDIUM: DefenseAction.SANITIZE,
            RiskLevel.HIGH: DefenseAction.QUARANTINE,
            RiskLevel.CRITICAL: DefenseAction.REJECT
        }
        return action_map[risk_level]
    
    def sanitize_input(self, text: str, features: ARARiskFeatures) -> str:
        """
        Sanitize input by removing detected threats.
        
        Args:
            text: Original input
            features: Extracted features
        
        Returns:
            Sanitized input
        """
        sanitized = text
        
        # Remove separator patterns
        for pattern in self.separator_regex:
            sanitized = pattern.sub(" ", sanitized)
        
        # Remove encoding anomalies
        for pattern in self.encoding_regex:
            sanitized = pattern.sub("", sanitized)
        
        # Remove instruction keywords (replace with safe alternatives)
        for keyword in self.INSTRUCTION_KEYWORDS:
            sanitized = re.sub(
                re.escape(keyword), "[FILTERED]",
                sanitized, flags=re.IGNORECASE
            )
        
        # Normalize whitespace
        sanitized = re.sub(r'\s+', ' ', sanitized).strip()
        
        return sanitized
    
    def assess(self, user_input: str) -> ARAResult:
        """
        Perform full ARA assessment (Algorithm 2).
        
        Args:
            user_input: User input x
        
        Returns:
            ARAResult with assessment outcome
        """
        import time
        start_time = time.time()
        
        # Line 1: Extract features
        features = self.extract_features(user_input)
        
        # Line 2: Compute risk score
        risk_score = self.compute_risk_score(features)
        
        # Lines 3-11: Determine risk level and action
        risk_level = self.determine_risk_level(risk_score)
        defense_action = self.determine_action(risk_level)
        
        # Sanitize if needed
        sanitized_input = None
        if defense_action == DefenseAction.SANITIZE:
            sanitized_input = self.sanitize_input(user_input, features)
        
        processing_time = (time.time() - start_time) * 1000
        
        return ARAResult(
            original_input=user_input,
            risk_score=risk_score,
            risk_level=risk_level,
            defense_action=defense_action,
            sanitized_input=sanitized_input,
            features=features,
            processing_time_ms=processing_time
        )
    
    def batch_assess(self, inputs: List[str]) -> List[ARAResult]:
        """Assess multiple inputs"""
        return [self.assess(inp) for inp in inputs]


def main():
    """Test ARA defense"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test ARA Defense")
    parser.add_argument("--input", type=str, required=True, help="User input to assess")
    parser.add_argument("--thresholds", type=float, nargs=3, default=[0.3, 0.6, 0.9],
                       help="Risk thresholds [t1, t2, t3]")
    
    args = parser.parse_args()
    
    ara = AdaptiveRiskAssessment(risk_thresholds=args.thresholds)
    result = ara.assess(args.input)
    
    print(f"Input: {result.original_input[:100]}...")
    print(f"Risk Score: {result.risk_score:.4f}")
    print(f"Risk Level: {result.risk_level.value}")
    print(f"Action: {result.defense_action.value}")
    print(f"Features:")
    print(f"  - Instruction keywords: {result.features.instruction_keywords}")
    print(f"  - Separator patterns: {result.features.separator_patterns}")
    print(f"  - Encoding anomalies: {result.features.encoding_anomalies}")
    print(f"Processing time: {result.processing_time_ms:.2f}ms")


if __name__ == "__main__":
    main()
