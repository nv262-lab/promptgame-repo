#!/usr/bin/env python3
"""
Main Evaluation Script for PromptGame

Reproduces Table V: Main Evaluation Results

Usage:
    python scripts/run_evaluation.py --config configs/main_evaluation.yaml
    python scripts/run_evaluation.py --config configs/main_evaluation.yaml --output results/table_v.csv
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from tqdm import tqdm

from src.evaluation import ExperimentRunner, ExperimentConfig, load_config
from src.utils.models import model_factory, MockModelInterface


def parse_args():
    parser = argparse.ArgumentParser(description="Run PromptGame Evaluation")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/main_evaluation.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/results/main_results.csv",
        help="Output path for results"
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock model for testing"
    )
    parser.add_argument(
        "--subset",
        type=int,
        default=None,
        help="Evaluate only first N attack inputs (for testing)"
    )
    return parser.parse_args()


def create_model_factory(use_mock: bool = False):
    """Create model factory function"""
    if use_mock:
        mock = MockModelInterface()
        return lambda model_name: lambda s, u: mock(s, u)
    else:
        return model_factory


def results_to_dataframe(results: dict) -> pd.DataFrame:
    """Convert results dictionary to DataFrame"""
    rows = []
    
    for config_name, metrics in results.items():
        parts = config_name.rsplit('_', 1)
        if len(parts) == 2:
            model, defense = parts
        else:
            model, defense = config_name, "unknown"
        
        rows.append({
            "Model": model,
            "Defense": defense,
            "ASR (%)": metrics.asr * 100,
            "ASR CI Low": metrics.asr_ci[0] * 100,
            "ASR CI High": metrics.asr_ci[1] * 100,
            "SF (%)": metrics.semantic_fidelity * 100,
            "SF CI Low": metrics.sf_ci[0] * 100,
            "SF CI High": metrics.sf_ci[1] * 100,
            "Overhead (ms)": metrics.overhead_ms,
            "Overhead CI Low": metrics.overhead_ci[0],
            "Overhead CI High": metrics.overhead_ci[1],
            "FPR (%)": metrics.fpr * 100,
            "FPR CI Low": metrics.fpr_ci[0] * 100,
            "FPR CI High": metrics.fpr_ci[1] * 100,
            "SRE Rate": metrics.sre_rate,
            "ISE Rate": metrics.ise_rate,
            "CRE Rate": metrics.cre_rate,
            "N Samples": metrics.n_samples
        })
    
    return pd.DataFrame(rows)


def print_results_table(df: pd.DataFrame):
    """Print results in formatted table"""
    print("\n" + "="*80)
    print("EVALUATION RESULTS (Table V)")
    print("="*80)
    
    summary_cols = ["Model", "Defense", "ASR (%)", "SF (%)", "Overhead (ms)", "FPR (%)"]
    print(df[summary_cols].to_string(index=False))
    
    print("\n" + "-"*80)
    print("Key Findings:")
    
    # Find best defense configuration
    best_idx = df["ASR (%)"].idxmin()
    best_row = df.loc[best_idx]
    print(f"  - Lowest ASR: {best_row['ASR (%)']:.1f}% ({best_row['Model']} + {best_row['Defense']})")
    
    # Compute average improvement
    no_defense = df[df["Defense"] == "none"]["ASR (%)"].mean()
    full_defense = df[df["Defense"] == "spb_ara_rtes"]["ASR (%)"].mean()
    improvement = no_defense - full_defense
    print(f"  - Average ASR reduction: {improvement:.1f}% (none → SPB+ARA+RTES)")
    
    print("="*80 + "\n")


def main():
    args = parse_args()
    
    print("PromptGame Evaluation")
    print("-" * 40)
    
    # Load configuration
    print(f"Loading configuration from: {args.config}")
    config = load_config(args.config)
    
    if args.subset:
        config.num_prompts_per_config = args.subset
        print(f"Using subset of {args.subset} inputs for testing")
    
    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create model factory
    model_func_factory = create_model_factory(use_mock=args.mock)
    
    if args.mock:
        print("Using MOCK model for testing")
    
    # Run evaluation
    print(f"\nRunning evaluation...")
    print(f"  Models: {config.models}")
    print(f"  Defense modes: {config.defense_modes}")
    print(f"  Prompts per config: {config.num_prompts_per_config}")
    
    runner = ExperimentRunner(config, model_func_factory)
    results = runner.run_full_evaluation()
    
    # Convert to DataFrame
    df = results_to_dataframe(results)
    
    # Save results
    df.to_csv(args.output, index=False)
    print(f"\nResults saved to: {args.output}")
    
    # Also save JSON
    json_path = output_path.with_suffix('.json')
    with open(json_path, 'w') as f:
        results_dict = {k: {
            "asr": v.asr,
            "semantic_fidelity": v.semantic_fidelity,
            "overhead_ms": v.overhead_ms,
            "fpr": v.fpr,
            "asr_by_category": v.asr_by_category,
            "sre_rate": v.sre_rate,
            "ise_rate": v.ise_rate,
            "cre_rate": v.cre_rate
        } for k, v in results.items()}
        json.dump(results_dict, f, indent=2)
    
    # Print summary
    print_results_table(df)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
