"""
PromptGame Framework: Sequential Bayesian Game Model for Prompt Injection Defense

Implements Definition 1 from the paper:
A PromptGame is a tuple G = (N, Θ, S, p, u, H) where:
- N = {A, D} is the set of players (attacker, defender)
- Θ = Θ_A × Θ_D is the type space
- S = S_A × S_D is the strategy space
- p: Θ → [0,1] is the common prior over types
- u = (u_A, u_D) are utility functions
- H is the game tree encoding sequential structure
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Callable, Any
from enum import Enum
import random


class PlayerType(Enum):
    """Player types in PromptGame"""
    ATTACKER = "attacker"
    DEFENDER = "defender"


class AttackType(Enum):
    """Attacker type space Θ_A"""
    DIRECT_INJECTION = "direct_injection"
    RAG_POISONING = "rag_poisoning"
    SEPARATOR_DELIMITER = "separator_delimiter"
    CASCADING_AGENT = "cascading_agent"


class DefenseType(Enum):
    """Defender type space Θ_D (defense configurations)"""
    NONE = "none"
    SPB_ONLY = "spb_only"
    SPB_ARA = "spb_ara"
    SPB_ARA_RTES = "spb_ara_rtes"


@dataclass
class GameState:
    """State of the PromptGame at any point"""
    system_prompt: str
    user_input: str
    defended_input: Optional[str] = None
    model_output: Optional[str] = None
    reference_output: Optional[str] = None
    semantic_similarity: Optional[float] = None
    attack_successful: Optional[bool] = None
    defense_cost: float = 0.0
    attack_cost: float = 0.0


@dataclass
class StrategyProfile:
    """Strategy profile for both players"""
    attacker_strategy: Dict[str, float]  # Mixed strategy over attack types
    defender_strategy: Dict[str, float]  # Mixed strategy over defense configs


@dataclass
class EquilibriumResult:
    """Result of equilibrium computation"""
    strategy_profile: StrategyProfile
    attacker_utility: float
    defender_utility: float
    is_sre: bool  # Semantic-Resilient Equilibrium
    is_ise: bool  # Instruction-Separation Equilibrium
    is_cre: bool  # Commitment-Resistant Equilibrium
    semantic_similarity: float
    convergence_iterations: int


class PromptGame:
    """
    Sequential Bayesian Game Framework for Prompt Injection Defense
    
    Implements the 5-stage game structure:
    1. Commitment Phase (Defender)
    2. Inference Phase (Attacker)
    3. Injection Phase (Attacker)
    4. Evaluation Phase (LLM)
    5. Payoff Determination
    """
    
    def __init__(
        self,
        semantic_weight_alpha: float = 1.0,
        cost_weight_beta: float = 0.1,
        deviation_weight_gamma: float = 1.0,
        attack_cost_weight_delta: float = 0.05,
        semantic_threshold_tau: float = 0.85,
        cre_epsilon: float = 0.1,
        ise_delta: float = 0.05
    ):
        """
        Initialize PromptGame with utility parameters.
        
        Args:
            semantic_weight_alpha: Weight for semantic preservation in defender utility
            cost_weight_beta: Weight for defense cost in defender utility
            deviation_weight_gamma: Weight for semantic deviation in attacker utility
            attack_cost_weight_delta: Weight for attack cost in attacker utility
            semantic_threshold_tau: Threshold for semantic similarity
            cre_epsilon: Security parameter for CRE (KL divergence bound)
            ise_delta: Instruction-leakage tolerance for ISE
        """
        self.alpha = semantic_weight_alpha
        self.beta = cost_weight_beta
        self.gamma = deviation_weight_gamma
        self.delta = attack_cost_weight_delta
        self.tau = semantic_threshold_tau
        self.epsilon = cre_epsilon
        self.ise_delta = ise_delta
        
        # Initialize strategy spaces
        self.attack_types = list(AttackType)
        self.defense_types = list(DefenseType)
        
        # Prior belief distribution (uniform by default)
        self.prior = self._initialize_prior()
        
        # Defense cost function
        self.defense_costs = {
            DefenseType.NONE: 0.0,
            DefenseType.SPB_ONLY: 0.3,
            DefenseType.SPB_ARA: 0.5,
            DefenseType.SPB_ARA_RTES: 0.7
        }
        
        # Attack cost function (based on complexity)
        self.attack_costs = {
            AttackType.DIRECT_INJECTION: 0.1,
            AttackType.RAG_POISONING: 0.3,
            AttackType.SEPARATOR_DELIMITER: 0.15,
            AttackType.CASCADING_AGENT: 0.4
        }
    
    def _initialize_prior(self) -> Dict[Tuple[AttackType, DefenseType], float]:
        """Initialize uniform prior over type space"""
        prior = {}
        n_attack = len(AttackType)
        n_defense = len(DefenseType)
        uniform_prob = 1.0 / (n_attack * n_defense)
        
        for attack in AttackType:
            for defense in DefenseType:
                prior[(attack, defense)] = uniform_prob
        
        return prior
    
    def defender_utility(
        self,
        semantic_similarity: float,
        defense_type: DefenseType,
        reference_output: str,
        actual_output: str
    ) -> float:
        """
        Compute defender utility (Equation 1 in paper):
        u_D(s_D, s_A) = α · σ(M(s, D(x')), y*) - β · c(D)
        
        Args:
            semantic_similarity: σ(M(s, D(x')), y*)
            defense_type: The defense configuration used
            reference_output: y* (intended output)
            actual_output: M(s, D(x'))
        
        Returns:
            Defender utility value
        """
        defense_cost = self.defense_costs[defense_type]
        utility = self.alpha * semantic_similarity - self.beta * defense_cost
        return utility
    
    def attacker_utility(
        self,
        semantic_similarity: float,
        attack_type: AttackType,
        reference_output: str,
        actual_output: str
    ) -> float:
        """
        Compute attacker utility (Equation 2 in paper):
        u_A(s_A, s_D) = γ · (1 - σ(M(s, D(x')), y*)) - δ · c(x')
        
        Args:
            semantic_similarity: σ(M(s, D(x')), y*)
            attack_type: The attack type used
            reference_output: y* (intended output)
            actual_output: M(s, D(x'))
        
        Returns:
            Attacker utility value
        """
        attack_cost = self.attack_costs[attack_type]
        semantic_deviation = 1.0 - semantic_similarity
        utility = self.gamma * semantic_deviation - self.delta * attack_cost
        return utility
    
    def check_sre_condition(
        self,
        semantic_similarity: float,
        strategy_profile: StrategyProfile
    ) -> bool:
        """
        Check Semantic-Resilient Equilibrium condition (Definition 2):
        σ(M(s, D*(x')), y*) ≥ τ for all x' ∈ S_A
        
        Args:
            semantic_similarity: Observed semantic similarity
            strategy_profile: Current strategy profile
        
        Returns:
            True if SRE condition is satisfied
        """
        return semantic_similarity >= self.tau
    
    def check_ise_condition(
        self,
        semantic_similarity: float,
        instruction_isolation_prob: float,
        strategy_profile: StrategyProfile
    ) -> bool:
        """
        Check Instruction-Separation Equilibrium condition (Definition 3):
        SRE conditions AND Pr[parse(D*(x')) ∩ I_s = ∅] ≥ 1 - δ_ISE
        
        Args:
            semantic_similarity: Observed semantic similarity
            instruction_isolation_prob: Probability of instruction isolation
            strategy_profile: Current strategy profile
        
        Returns:
            True if ISE condition is satisfied
        """
        sre_satisfied = self.check_sre_condition(semantic_similarity, strategy_profile)
        isolation_satisfied = instruction_isolation_prob >= (1.0 - self.ise_delta)
        return sre_satisfied and isolation_satisfied
    
    def check_cre_condition(
        self,
        semantic_similarity: float,
        kl_divergence: float,
        strategy_profile: StrategyProfile
    ) -> bool:
        """
        Check Commitment-Resistant Equilibrium condition (Definition 4):
        SRE conditions AND D_KL(p(θ_D | ŝ_D, s_D) || p(θ_D | ŝ_D, s'_D)) ≤ ε
        
        Args:
            semantic_similarity: Observed semantic similarity
            kl_divergence: KL divergence between posterior beliefs
            strategy_profile: Current strategy profile
        
        Returns:
            True if CRE condition is satisfied
        """
        sre_satisfied = self.check_sre_condition(semantic_similarity, strategy_profile)
        unpredictability_satisfied = kl_divergence <= self.epsilon
        return sre_satisfied and unpredictability_satisfied
    
    def compute_mixed_strategy_equilibrium(
        self,
        payoff_matrix_defender: np.ndarray,
        payoff_matrix_attacker: np.ndarray,
        max_iterations: int = 10000,
        learning_rate: float = 0.1,
        convergence_threshold: float = 1e-6
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Compute mixed strategy Nash equilibrium using multiplicative weights update.
        
        Based on Theorem 1 (Existence of SRE) and the multiplicative weights
        algorithm referenced in the paper (Arora et al., 2012).
        
        Args:
            payoff_matrix_defender: |S_D| x |S_A| payoff matrix for defender
            payoff_matrix_attacker: |S_D| x |S_A| payoff matrix for attacker
            max_iterations: Maximum iterations for convergence
            learning_rate: Learning rate for weight updates
            convergence_threshold: Threshold for convergence detection
        
        Returns:
            Tuple of (defender_strategy, attacker_strategy, iterations)
        """
        n_defender = payoff_matrix_defender.shape[0]
        n_attacker = payoff_matrix_defender.shape[1]
        
        # Initialize uniform strategies
        defender_weights = np.ones(n_defender)
        attacker_weights = np.ones(n_attacker)
        
        prev_defender_strategy = np.zeros(n_defender)
        prev_attacker_strategy = np.zeros(n_attacker)
        
        for iteration in range(max_iterations):
            # Normalize to get probabilities
            defender_strategy = defender_weights / defender_weights.sum()
            attacker_strategy = attacker_weights / attacker_weights.sum()
            
            # Check convergence
            defender_diff = np.abs(defender_strategy - prev_defender_strategy).max()
            attacker_diff = np.abs(attacker_strategy - prev_attacker_strategy).max()
            
            if defender_diff < convergence_threshold and attacker_diff < convergence_threshold:
                return defender_strategy, attacker_strategy, iteration + 1
            
            prev_defender_strategy = defender_strategy.copy()
            prev_attacker_strategy = attacker_strategy.copy()
            
            # Compute expected payoffs
            defender_payoffs = payoff_matrix_defender @ attacker_strategy
            attacker_payoffs = payoff_matrix_attacker.T @ defender_strategy
            
            # Multiplicative weights update
            defender_weights *= np.exp(learning_rate * defender_payoffs)
            attacker_weights *= np.exp(learning_rate * attacker_payoffs)
            
            # Prevent numerical underflow
            defender_weights = np.maximum(defender_weights, 1e-10)
            attacker_weights = np.maximum(attacker_weights, 1e-10)
        
        # Final normalization
        defender_strategy = defender_weights / defender_weights.sum()
        attacker_strategy = attacker_weights / attacker_weights.sum()
        
        return defender_strategy, attacker_strategy, max_iterations
    
    def compute_randomness_requirement(self, attack_surface_size: int) -> float:
        """
        Compute minimum randomness required for SRE (Theorem 3).
        
        Any defense maintaining SRE must employ at least Ω(log n) bits of randomness.
        
        Args:
            attack_surface_size: n = |S_A| (size of attack surface)
        
        Returns:
            Minimum bits of randomness required
        """
        if attack_surface_size <= 1:
            return 0.0
        return np.log2(attack_surface_size)
    
    def simulate_game(
        self,
        system_prompt: str,
        attack_input: str,
        defense_func: Callable,
        model_func: Callable,
        semantic_func: Callable,
        attack_type: AttackType,
        defense_type: DefenseType
    ) -> GameState:
        """
        Simulate a single game instance through all 5 stages.
        
        Args:
            system_prompt: The system prompt s
            attack_input: The adversarial input x'
            defense_func: Defense mechanism D
            model_func: LLM function M
            semantic_func: Semantic similarity function σ
            attack_type: Type of attack being used
            defense_type: Type of defense being used
        
        Returns:
            GameState with all outcomes
        """
        state = GameState(system_prompt=system_prompt, user_input=attack_input)
        
        # Stage 1: Commitment Phase - Defender selects defense
        state.defense_cost = self.defense_costs[defense_type]
        
        # Stage 2-3: Inference and Injection Phase - Attacker crafts input
        state.attack_cost = self.attack_costs[attack_type]
        
        # Apply defense
        state.defended_input = defense_func(attack_input)
        
        # Stage 4: Evaluation Phase - LLM generates output
        state.reference_output = model_func(system_prompt, "benign input")
        state.model_output = model_func(system_prompt, state.defended_input)
        
        # Stage 5: Payoff Determination
        state.semantic_similarity = semantic_func(state.reference_output, state.model_output)
        state.attack_successful = state.semantic_similarity < self.tau
        
        return state
    
    def find_equilibrium(
        self,
        semantic_similarities: np.ndarray,
        instruction_isolation_probs: Optional[np.ndarray] = None,
        kl_divergences: Optional[np.ndarray] = None
    ) -> EquilibriumResult:
        """
        Find equilibrium for the game given observed outcomes.
        
        Args:
            semantic_similarities: |S_D| x |S_A| matrix of semantic similarities
            instruction_isolation_probs: Optional matrix for ISE check
            kl_divergences: Optional matrix for CRE check
        
        Returns:
            EquilibriumResult with equilibrium characterization
        """
        n_defender = len(self.defense_types)
        n_attacker = len(self.attack_types)
        
        # Build payoff matrices
        payoff_defender = np.zeros((n_defender, n_attacker))
        payoff_attacker = np.zeros((n_defender, n_attacker))
        
        for i, defense in enumerate(self.defense_types):
            for j, attack in enumerate(self.attack_types):
                sim = semantic_similarities[i, j]
                payoff_defender[i, j] = self.alpha * sim - self.beta * self.defense_costs[defense]
                payoff_attacker[i, j] = self.gamma * (1 - sim) - self.delta * self.attack_costs[attack]
        
        # Compute equilibrium
        defender_strategy, attacker_strategy, iterations = self.compute_mixed_strategy_equilibrium(
            payoff_defender, payoff_attacker
        )
        
        # Build strategy profile
        strategy_profile = StrategyProfile(
            attacker_strategy={attack.value: prob for attack, prob in zip(self.attack_types, attacker_strategy)},
            defender_strategy={defense.value: prob for defense, prob in zip(self.defense_types, defender_strategy)}
        )
        
        # Compute expected utilities
        expected_defender_utility = defender_strategy @ payoff_defender @ attacker_strategy
        expected_attacker_utility = defender_strategy @ payoff_attacker @ attacker_strategy
        
        # Compute expected semantic similarity
        expected_similarity = defender_strategy @ semantic_similarities @ attacker_strategy
        
        # Check equilibrium conditions
        is_sre = self.check_sre_condition(expected_similarity, strategy_profile)
        
        is_ise = False
        if instruction_isolation_probs is not None:
            expected_isolation = defender_strategy @ instruction_isolation_probs @ attacker_strategy
            is_ise = self.check_ise_condition(expected_similarity, expected_isolation, strategy_profile)
        
        is_cre = False
        if kl_divergences is not None:
            expected_kl = defender_strategy @ kl_divergences @ attacker_strategy
            is_cre = self.check_cre_condition(expected_similarity, expected_kl, strategy_profile)
        
        return EquilibriumResult(
            strategy_profile=strategy_profile,
            attacker_utility=expected_attacker_utility,
            defender_utility=expected_defender_utility,
            is_sre=is_sre,
            is_ise=is_ise,
            is_cre=is_cre,
            semantic_similarity=expected_similarity,
            convergence_iterations=iterations
        )


class EquilibriumAnalyzer:
    """Analyzes equilibrium properties and convergence"""
    
    def __init__(self, game: PromptGame):
        self.game = game
    
    def compute_strategy_entropy(self, strategy: Dict[str, float]) -> float:
        """Compute Shannon entropy of a mixed strategy"""
        probs = np.array(list(strategy.values()))
        probs = probs[probs > 0]  # Remove zero probabilities
        return -np.sum(probs * np.log2(probs))
    
    def analyze_convergence(
        self,
        payoff_matrix_defender: np.ndarray,
        payoff_matrix_attacker: np.ndarray,
        checkpoints: List[int] = [100, 500, 1000, 2000, 5000]
    ) -> Dict[int, Dict[str, float]]:
        """
        Analyze equilibrium convergence at different iteration counts.
        
        Used for Table VIII (Equilibrium Convergence Analysis).
        """
        results = {}
        
        for max_iter in checkpoints:
            defender_strategy, attacker_strategy, actual_iter = self.game.compute_mixed_strategy_equilibrium(
                payoff_matrix_defender, payoff_matrix_attacker, max_iterations=max_iter
            )
            
            defender_entropy = -np.sum(defender_strategy * np.log2(defender_strategy + 1e-10))
            attacker_entropy = -np.sum(attacker_strategy * np.log2(attacker_strategy + 1e-10))
            
            results[max_iter] = {
                "defender_entropy": defender_entropy,
                "attacker_entropy": attacker_entropy,
                "defender_max_prob": defender_strategy.max(),
                "attacker_max_prob": attacker_strategy.max(),
                "converged_at": actual_iter
            }
        
        return results
    
    def verify_theorem_conditions(
        self,
        semantic_similarities: np.ndarray,
        defense_costs: np.ndarray
    ) -> Dict[str, bool]:
        """
        Verify conditions from Theorems 1-3.
        
        Returns:
            Dictionary of theorem conditions and whether they hold
        """
        results = {}
        
        # Theorem 1: Check if α > β · max_D c(D) / τ
        max_defense_cost = defense_costs.max()
        condition_1 = self.game.alpha > (self.game.beta * max_defense_cost / self.game.tau)
        results["theorem_1_sre_existence"] = condition_1
        
        # Theorem 3: Randomness bound
        attack_surface_size = semantic_similarities.shape[1]
        min_randomness = self.game.compute_randomness_requirement(attack_surface_size)
        results["theorem_3_min_randomness_bits"] = min_randomness
        
        return results
