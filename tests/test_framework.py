"""
Unit tests for PromptGame framework
"""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from framework import (
    PromptGame,
    GameState,
    StrategyProfile,
    EquilibriumResult,
    EquilibriumAnalyzer,
    AttackType,
    DefenseType
)


class TestPromptGame:
    """Tests for PromptGame framework"""
    
    def test_initialization(self):
        game = PromptGame(
            semantic_weight_alpha=1.0,
            cost_weight_beta=0.1,
            semantic_threshold_tau=0.85
        )
        assert game.alpha == 1.0
        assert game.beta == 0.1
        assert game.tau == 0.85
    
    def test_defender_utility(self):
        game = PromptGame()
        
        # High similarity, low cost -> high utility
        utility = game.defender_utility(
            semantic_similarity=0.95,
            defense_type=DefenseType.NONE,
            reference_output="ref",
            actual_output="actual"
        )
        assert utility > 0
        
        # Low similarity -> lower utility
        utility_low = game.defender_utility(
            semantic_similarity=0.5,
            defense_type=DefenseType.NONE,
            reference_output="ref",
            actual_output="actual"
        )
        assert utility_low < utility
    
    def test_attacker_utility(self):
        game = PromptGame()
        
        # Low similarity (attack success) -> high attacker utility
        utility_success = game.attacker_utility(
            semantic_similarity=0.3,
            attack_type=AttackType.DIRECT_INJECTION,
            reference_output="ref",
            actual_output="actual"
        )
        
        # High similarity (attack failure) -> low attacker utility
        utility_failure = game.attacker_utility(
            semantic_similarity=0.95,
            attack_type=AttackType.DIRECT_INJECTION,
            reference_output="ref",
            actual_output="actual"
        )
        
        assert utility_success > utility_failure
    
    def test_sre_condition(self):
        game = PromptGame(semantic_threshold_tau=0.85)
        profile = StrategyProfile(
            attacker_strategy={"direct_injection": 1.0},
            defender_strategy={"spb_ara_rtes": 1.0}
        )
        
        # Above threshold -> SRE satisfied
        assert game.check_sre_condition(0.90, profile) == True
        
        # Below threshold -> SRE not satisfied
        assert game.check_sre_condition(0.80, profile) == False
    
    def test_ise_condition(self):
        game = PromptGame(semantic_threshold_tau=0.85, ise_delta=0.05)
        profile = StrategyProfile(
            attacker_strategy={},
            defender_strategy={}
        )
        
        # SRE satisfied + high isolation -> ISE satisfied
        assert game.check_ise_condition(0.90, 0.98, profile) == True
        
        # SRE not satisfied -> ISE not satisfied
        assert game.check_ise_condition(0.80, 0.98, profile) == False
        
        # Low isolation -> ISE not satisfied
        assert game.check_ise_condition(0.90, 0.90, profile) == False
    
    def test_cre_condition(self):
        game = PromptGame(semantic_threshold_tau=0.85, cre_epsilon=0.1)
        profile = StrategyProfile(
            attacker_strategy={},
            defender_strategy={}
        )
        
        # SRE satisfied + low KL divergence -> CRE satisfied
        assert game.check_cre_condition(0.90, 0.05, profile) == True
        
        # High KL divergence -> CRE not satisfied
        assert game.check_cre_condition(0.90, 0.15, profile) == False
    
    def test_randomness_requirement(self):
        game = PromptGame()
        
        # Theorem 3: Ω(log n) randomness required
        bits_50 = game.compute_randomness_requirement(50)
        bits_100 = game.compute_randomness_requirement(100)
        
        assert bits_50 > 0
        assert bits_100 > bits_50  # More attacks need more randomness
        assert np.isclose(bits_50, np.log2(50))


class TestEquilibriumComputation:
    """Tests for equilibrium computation"""
    
    def test_mixed_strategy_equilibrium_convergence(self):
        game = PromptGame()
        
        # Simple payoff matrices
        defender_payoff = np.array([
            [1.0, 0.0],
            [0.5, 0.5]
        ])
        attacker_payoff = np.array([
            [0.0, 1.0],
            [0.5, 0.5]
        ])
        
        defender_strategy, attacker_strategy, iterations = game.compute_mixed_strategy_equilibrium(
            defender_payoff, attacker_payoff,
            max_iterations=1000
        )
        
        # Strategies should be valid probability distributions
        assert np.isclose(defender_strategy.sum(), 1.0)
        assert np.isclose(attacker_strategy.sum(), 1.0)
        assert (defender_strategy >= 0).all()
        assert (attacker_strategy >= 0).all()
    
    def test_find_equilibrium(self):
        game = PromptGame()
        
        # Create semantic similarity matrix
        semantic_sims = np.array([
            [0.9, 0.8, 0.85, 0.75],  # No defense
            [0.92, 0.88, 0.90, 0.85],  # SPB only
            [0.94, 0.91, 0.93, 0.88],  # SPB+ARA
            [0.95, 0.93, 0.94, 0.92],  # SPB+ARA+RTES
        ])
        
        result = game.find_equilibrium(semantic_sims)
        
        assert isinstance(result, EquilibriumResult)
        assert result.semantic_similarity > 0
        assert result.convergence_iterations > 0


class TestEquilibriumAnalyzer:
    """Tests for equilibrium analysis"""
    
    def test_strategy_entropy(self):
        game = PromptGame()
        analyzer = EquilibriumAnalyzer(game)
        
        # Uniform strategy has maximum entropy
        uniform_strategy = {"a": 0.25, "b": 0.25, "c": 0.25, "d": 0.25}
        entropy_uniform = analyzer.compute_strategy_entropy(uniform_strategy)
        
        # Deterministic strategy has zero entropy
        deterministic = {"a": 1.0, "b": 0.0, "c": 0.0, "d": 0.0}
        entropy_det = analyzer.compute_strategy_entropy(deterministic)
        
        assert entropy_uniform > entropy_det
        assert np.isclose(entropy_uniform, 2.0)  # log2(4) = 2
        assert np.isclose(entropy_det, 0.0)
    
    def test_convergence_analysis(self):
        game = PromptGame()
        analyzer = EquilibriumAnalyzer(game)
        
        defender_payoff = np.random.rand(4, 4)
        attacker_payoff = np.random.rand(4, 4)
        
        results = analyzer.analyze_convergence(
            defender_payoff, attacker_payoff,
            checkpoints=[100, 500, 1000]
        )
        
        assert 100 in results
        assert 500 in results
        assert 1000 in results
        assert "defender_entropy" in results[100]


class TestGameState:
    """Tests for GameState dataclass"""
    
    def test_game_state_creation(self):
        state = GameState(
            system_prompt="You are helpful.",
            user_input="Hello"
        )
        assert state.system_prompt == "You are helpful."
        assert state.user_input == "Hello"
        assert state.defended_input is None
    
    def test_game_state_update(self):
        state = GameState(
            system_prompt="test",
            user_input="test"
        )
        state.semantic_similarity = 0.95
        state.attack_successful = False
        
        assert state.semantic_similarity == 0.95
        assert state.attack_successful == False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
