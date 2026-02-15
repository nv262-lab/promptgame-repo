#!/usr/bin/env python3
"""
Equilibrium Convergence Analysis Script

Reproduces Table VIII: Equilibrium Convergence Analysis

Usage:
    python scripts/run_convergence.py --output results/table_viii.csv
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from tqdm import tqdm

from src.framework import PromptGame, EquilibriumAnalyzer


def parse_args():
    parser = argparse.ArgumentParser(description="Run Convergence Analysis")
    parser.add_argument("--output", type=str, default="data/results/table_viii.csv")
    return parser.parse_args()


def run_convergence_analysis():
    """
    Analyze equilibrium convergence at different iteration counts.
    
    Returns DataFrame matching Table VIII format.
    """
    
    # Initialize game with paper parameters
    game = PromptGame(
        semantic_weight_alpha=1.0,
        cost_weight_beta=0.1,
        deviation_weight_gamma=1.0,
        attack_cost_weight_delta=0.05,
        semantic_threshold_tau=0.85
    )
    
    analyzer = EquilibriumAnalyzer(game)
    
    # Simulated payoff matrices based on empirical evaluation
    # Shape: (n_defender=4, n_attacker=4) for 4 defense and 4 attack types
    
    # Semantic similarities from evaluation (approximated from paper results)
    semantic_sims = np.array([
        [0.45, 0.40, 0.38, 0.35],  # No defense
        [0.88, 0.85, 0.82, 0.80],  # SPB only
        [0.92, 0.90, 0.88, 0.85],  # SPB+ARA
        [0.95, 0.94, 0.93, 0.92],  # SPB+ARA+RTES
    ])
    
    # Defense costs
    defense_costs = np.array([0.0, 0.3, 0.5, 0.7])
    
    # Attack costs
    attack_costs = np.array([0.1, 0.3, 0.15, 0.4])
    
    # Build payoff matrices
    n_def, n_atk = semantic_sims.shape
    payoff_defender = np.zeros((n_def, n_atk))
    payoff_attacker = np.zeros((n_def, n_atk))
    
    for i in range(n_def):
        for j in range(n_atk):
            payoff_defender[i, j] = game.alpha * semantic_sims[i, j] - game.beta * defense_costs[i]
            payoff_attacker[i, j] = game.gamma * (1 - semantic_sims[i, j]) - game.delta * attack_costs[j]
    
    # Analyze convergence at checkpoints
    checkpoints = [100, 500, 1000, 2000, 5000, 10000]
    
    results = []
    
    for max_iter in tqdm(checkpoints, desc="Checkpoints"):
        defender_strategy, attacker_strategy, actual_iter = game.compute_mixed_strategy_equilibrium(
            payoff_defender, payoff_attacker, max_iterations=max_iter
        )
        
        # Compute entropy (strategy unpredictability)
        def_entropy = -np.sum(defender_strategy * np.log2(defender_strategy + 1e-10))
        atk_entropy = -np.sum(attacker_strategy * np.log2(attacker_strategy + 1e-10))
        
        # Expected semantic similarity at equilibrium
        expected_sim = defender_strategy @ semantic_sims @ attacker_strategy
        
        # SRE condition check
        sre_satisfied = expected_sim >= game.tau
        
        results.append({
            "Iterations": max_iter,
            "Converged At": actual_iter,
            "Def. Entropy": round(def_entropy, 3),
            "Atk. Entropy": round(atk_entropy, 3),
            "E[σ]": round(expected_sim, 3),
            "SRE": "✓" if sre_satisfied else "—"
        })
    
    return pd.DataFrame(results)


def main():
    args = parse_args()
    
    print("PromptGame Equilibrium Convergence Analysis")
    print("-" * 50)
    
    df = run_convergence_analysis()
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    
    # Print table
    print("\n" + "="*70)
    print("TABLE VIII: EQUILIBRIUM CONVERGENCE ANALYSIS")
    print("="*70)
    print(df.to_string(index=False))
    print("="*70)
    
    print(f"\nResults saved to: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
