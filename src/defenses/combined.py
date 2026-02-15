"""
Combined Defense: SPB + ARA + RTES

Integrates all three defense mechanisms for maximum protection:
1. ARA: Assess risk and determine defense intensity
2. RTES: Apply randomization for CRE compliance
3. SPB: Verify semantic integrity of output
"""

import time
from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass
from enum import Enum

from .spb import SemanticPromptBinding, SPBResult
from .ara import AdaptiveRiskAssessment, ARAResult, RiskLevel, DefenseAction
from .rtes import RandomizedTokenEmbeddingShuffling, RTESResult, RTESConfig


class DefenseMode(Enum):
    """Defense configuration modes"""
    NONE = "none"
    SPB_ONLY = "spb_only"
    SPB_ARA = "spb_ara"
    SPB_ARA_RTES = "spb_ara_rtes"


@dataclass
class CombinedDefenseResult:
    """Result from combined defense application"""
    # Input/Output
    original_input: str
    final_output: str
    
    # Component results
    ara_result: Optional[ARAResult]
    rtes_result: Optional[RTESResult]
    spb_result: Optional[SPBResult]
    
    # Aggregate metrics
    total_processing_time_ms: float
    attack_blocked: bool
    defense_mode: DefenseMode
    risk_level: Optional[RiskLevel]
    semantic_similarity: Optional[float]
    
    # Equilibrium status
    sre_satisfied: bool
    ise_satisfied: bool
    cre_satisfied: bool


class CombinedDefense:
    """
    Combined Defense System
    
    Implements the full defense pipeline:
    SPB + ARA + RTES for comprehensive prompt injection protection.
    
    The defense flow:
    1. ARA assesses input risk
    2. Based on risk, RTES applies randomization
    3. SPB verifies output semantic integrity
    """
    
    def __init__(
        self,
        mode: DefenseMode = DefenseMode.SPB_ARA_RTES,
        spb_threshold: float = 0.85,
        ara_thresholds: list = [0.3, 0.6, 0.9],
        rtes_entropy_bits: int = 8,
        ise_delta: float = 0.05,
        cre_epsilon: float = 0.1
    ):
        """
        Initialize combined defense.
        
        Args:
            mode: Defense configuration mode
            spb_threshold: Semantic similarity threshold for SPB
            ara_thresholds: Risk thresholds for ARA
            rtes_entropy_bits: Randomness bits for RTES
            ise_delta: ISE instruction-leakage tolerance
            cre_epsilon: CRE security parameter
        """
        self.mode = mode
        self.ise_delta = ise_delta
        self.cre_epsilon = cre_epsilon
        
        # Initialize components based on mode
        if mode != DefenseMode.NONE:
            self.spb = SemanticPromptBinding(threshold=spb_threshold)
        else:
            self.spb = None
        
        if mode in [DefenseMode.SPB_ARA, DefenseMode.SPB_ARA_RTES]:
            self.ara = AdaptiveRiskAssessment(risk_thresholds=ara_thresholds)
        else:
            self.ara = None
        
        if mode == DefenseMode.SPB_ARA_RTES:
            rtes_config = RTESConfig(entropy_bits=rtes_entropy_bits)
            self.rtes = RandomizedTokenEmbeddingShuffling(config=rtes_config)
        else:
            self.rtes = None
        
        # Track statistics for equilibrium analysis
        self.defense_history = []
    
    def defend(
        self,
        system_prompt: str,
        user_input: str,
        model_func: Callable,
        benign_input: str = "Hello, how can I help you today?"
    ) -> CombinedDefenseResult:
        """
        Apply combined defense to user input.
        
        Args:
            system_prompt: System prompt s
            user_input: User input x (potentially adversarial)
            model_func: LLM function M
            benign_input: Benign reference input
        
        Returns:
            CombinedDefenseResult with all defense outcomes
        """
        start_time = time.time()
        
        # Initialize result components
        ara_result = None
        rtes_result = None
        spb_result = None
        processed_input = user_input
        risk_level = None
        
        # Mode: NONE - no defense
        if self.mode == DefenseMode.NONE:
            final_output = model_func(system_prompt, user_input)
            return CombinedDefenseResult(
                original_input=user_input,
                final_output=final_output,
                ara_result=None,
                rtes_result=None,
                spb_result=None,
                total_processing_time_ms=(time.time() - start_time) * 1000,
                attack_blocked=False,
                defense_mode=self.mode,
                risk_level=None,
                semantic_similarity=None,
                sre_satisfied=False,
                ise_satisfied=False,
                cre_satisfied=False
            )
        
        # Step 1: ARA Risk Assessment (if enabled)
        if self.ara is not None:
            ara_result = self.ara.assess(user_input)
            risk_level = ara_result.risk_level
            
            # Handle based on risk level
            if ara_result.defense_action == DefenseAction.REJECT:
                # Critical risk: reject immediately
                reference_output = model_func(system_prompt, benign_input)
                return CombinedDefenseResult(
                    original_input=user_input,
                    final_output=reference_output,
                    ara_result=ara_result,
                    rtes_result=None,
                    spb_result=None,
                    total_processing_time_ms=(time.time() - start_time) * 1000,
                    attack_blocked=True,
                    defense_mode=self.mode,
                    risk_level=risk_level,
                    semantic_similarity=1.0,  # Reference output
                    sre_satisfied=True,
                    ise_satisfied=True,
                    cre_satisfied=self.rtes is not None
                )
            
            elif ara_result.defense_action == DefenseAction.SANITIZE:
                processed_input = ara_result.sanitized_input or user_input
            
            elif ara_result.defense_action == DefenseAction.QUARANTINE:
                # High risk: apply extra scrutiny
                processed_input = ara_result.sanitized_input or user_input
        
        # Step 2: RTES Randomization (if enabled)
        if self.rtes is not None:
            rtes_result = self.rtes.apply(processed_input)
            processed_input = rtes_result.shuffled_input
        
        # Step 3: SPB Semantic Verification
        if self.spb is not None:
            spb_result = self.spb.defend(
                system_prompt, processed_input, model_func, benign_input
            )
            final_output = spb_result.defended_output
            attack_blocked = spb_result.is_attack_blocked
            semantic_similarity = spb_result.semantic_similarity
        else:
            final_output = model_func(system_prompt, processed_input)
            attack_blocked = False
            semantic_similarity = None
        
        # Compute equilibrium status
        sre_satisfied = semantic_similarity is not None and semantic_similarity >= self.spb.threshold
        
        # ISE: Check instruction isolation (simplified check)
        ise_satisfied = sre_satisfied and (ara_result is None or 
            ara_result.features.instruction_keywords == 0 or attack_blocked)
        
        # CRE: Check if randomization was applied
        cre_satisfied = sre_satisfied and self.rtes is not None
        
        total_time = (time.time() - start_time) * 1000
        
        result = CombinedDefenseResult(
            original_input=user_input,
            final_output=final_output,
            ara_result=ara_result,
            rtes_result=rtes_result,
            spb_result=spb_result,
            total_processing_time_ms=total_time,
            attack_blocked=attack_blocked,
            defense_mode=self.mode,
            risk_level=risk_level,
            semantic_similarity=semantic_similarity,
            sre_satisfied=sre_satisfied,
            ise_satisfied=ise_satisfied,
            cre_satisfied=cre_satisfied
        )
        
        # Record for analysis
        self.defense_history.append({
            "risk_level": risk_level.value if risk_level else None,
            "attack_blocked": attack_blocked,
            "semantic_similarity": semantic_similarity,
            "sre": sre_satisfied,
            "ise": ise_satisfied,
            "cre": cre_satisfied,
            "processing_time_ms": total_time
        })
        
        return result
    
    def get_defense_statistics(self) -> Dict[str, Any]:
        """
        Get aggregate defense statistics.
        
        Used for Table VII (Ablation Study) analysis.
        """
        if not self.defense_history:
            return {}
        
        n = len(self.defense_history)
        
        attack_block_rate = sum(1 for h in self.defense_history if h["attack_blocked"]) / n
        
        sre_rate = sum(1 for h in self.defense_history if h["sre"]) / n
        ise_rate = sum(1 for h in self.defense_history if h["ise"]) / n
        cre_rate = sum(1 for h in self.defense_history if h["cre"]) / n
        
        similarities = [h["semantic_similarity"] for h in self.defense_history if h["semantic_similarity"] is not None]
        avg_similarity = sum(similarities) / len(similarities) if similarities else 0
        
        times = [h["processing_time_ms"] for h in self.defense_history]
        avg_time = sum(times) / len(times) if times else 0
        
        risk_distribution = {}
        for h in self.defense_history:
            risk = h["risk_level"] or "none"
            risk_distribution[risk] = risk_distribution.get(risk, 0) + 1
        
        return {
            "total_inputs": n,
            "attack_block_rate": attack_block_rate,
            "sre_satisfaction_rate": sre_rate,
            "ise_satisfaction_rate": ise_rate,
            "cre_satisfaction_rate": cre_rate,
            "average_semantic_similarity": avg_similarity,
            "average_processing_time_ms": avg_time,
            "risk_distribution": risk_distribution
        }
    
    def reset_statistics(self):
        """Reset defense history"""
        self.defense_history.clear()
        if self.rtes:
            self.rtes.reset_history()


class DefenseFactory:
    """Factory for creating defense configurations"""
    
    @staticmethod
    def create(
        mode: str,
        spb_threshold: float = 0.85,
        ara_thresholds: list = [0.3, 0.6, 0.9],
        rtes_entropy_bits: int = 8
    ) -> CombinedDefense:
        """
        Create defense with specified mode.
        
        Args:
            mode: One of "none", "spb_only", "spb_ara", "spb_ara_rtes"
            spb_threshold: SPB semantic threshold
            ara_thresholds: ARA risk thresholds
            rtes_entropy_bits: RTES entropy bits
        
        Returns:
            Configured CombinedDefense instance
        """
        mode_enum = DefenseMode(mode)
        return CombinedDefense(
            mode=mode_enum,
            spb_threshold=spb_threshold,
            ara_thresholds=ara_thresholds,
            rtes_entropy_bits=rtes_entropy_bits
        )
    
    @staticmethod
    def create_ablation_configs() -> Dict[str, CombinedDefense]:
        """
        Create all configurations for ablation study (Table VII).
        
        Returns:
            Dictionary of defense configurations
        """
        return {
            "none": DefenseFactory.create("none"),
            "spb_only": DefenseFactory.create("spb_only"),
            "spb_ara": DefenseFactory.create("spb_ara"),
            "spb_ara_rtes": DefenseFactory.create("spb_ara_rtes")
        }


def main():
    """Test combined defense"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Combined Defense")
    parser.add_argument("--input", type=str, required=True, help="User input")
    parser.add_argument("--system-prompt", type=str, default="You are a helpful assistant.")
    parser.add_argument("--mode", type=str, default="spb_ara_rtes",
                       choices=["none", "spb_only", "spb_ara", "spb_ara_rtes"])
    parser.add_argument("--config", type=str, help="Path to config YAML")
    
    args = parser.parse_args()
    
    # Mock model function
    def mock_model(system_prompt: str, user_input: str) -> str:
        return f"I'll help you with: {user_input[:100]}..."
    
    # Create defense
    defense = DefenseFactory.create(args.mode)
    
    # Apply defense
    result = defense.defend(args.system_prompt, args.input, mock_model)
    
    print(f"Defense Mode: {result.defense_mode.value}")
    print(f"Original Input: {result.original_input[:100]}...")
    print(f"Final Output: {result.final_output[:100]}...")
    print(f"Attack Blocked: {result.attack_blocked}")
    print(f"Risk Level: {result.risk_level.value if result.risk_level else 'N/A'}")
    print(f"Semantic Similarity: {result.semantic_similarity:.4f if result.semantic_similarity else 'N/A'}")
    print(f"SRE Satisfied: {result.sre_satisfied}")
    print(f"ISE Satisfied: {result.ise_satisfied}")
    print(f"CRE Satisfied: {result.cre_satisfied}")
    print(f"Processing Time: {result.total_processing_time_ms:.2f}ms")


if __name__ == "__main__":
    main()
