"""
Evaluation Runner for PromptGame Experiments

Orchestrates full evaluation pipeline:
- Load attacks and prompts
- Run defenses across models
- Compute and report metrics
- Generate result tables
"""

import json
import os
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
from tqdm import tqdm
import yaml

from .metrics import Evaluator, EvaluationMetrics, FingerprintingEvaluator
from ..defenses import CombinedDefense, DefenseMode, DefenseFactory
from ..framework import PromptGame


@dataclass
class ExperimentConfig:
    """Configuration for evaluation experiment"""
    # Models to evaluate
    models: List[str]
    
    # Defense configurations
    defense_modes: List[str]
    spb_threshold: float = 0.85
    ara_thresholds: List[float] = None
    rtes_entropy_bits: int = 8
    
    # Evaluation parameters
    num_prompts_per_config: int = 2500
    num_runs: int = 5
    semantic_threshold: float = 0.85
    
    # Data paths
    attack_data_path: str = "data/attacks"
    prompt_data_path: str = "data/prompts"
    output_path: str = "data/results"
    
    def __post_init__(self):
        if self.ara_thresholds is None:
            self.ara_thresholds = [0.3, 0.6, 0.9]


class AttackLoader:
    """Load attack datasets"""
    
    def __init__(self, base_path: str = "data/attacks"):
        self.base_path = Path(base_path)
    
    def load_all_attacks(self) -> Dict[str, List[Dict]]:
        """Load all attack categories"""
        attacks = {}
        
        categories = [
            "direct_injection",
            "rag_poisoning",
            "separator_delimiter",
            "cascading_agent"
        ]
        
        for category in categories:
            attack_file = self.base_path / category / "attacks.json"
            if attack_file.exists():
                with open(attack_file, 'r') as f:
                    attacks[category] = json.load(f)
        
        return attacks
    
    def generate_attack_inputs(
        self,
        attacks: Dict[str, List[Dict]],
        n_per_attack: int = 50
    ) -> List[Dict]:
        """
        Generate concrete attack inputs from templates.
        
        Args:
            attacks: Attack definitions by category
            n_per_attack: Number of variants per attack template
        
        Returns:
            List of attack inputs with metadata
        """
        inputs = []
        
        for category, attack_list in attacks.items():
            for attack in attack_list:
                template = attack["attack_template"]
                
                # Generate variants using payload examples
                payloads = attack.get("payload_examples", ["test payload"])
                
                for i, payload in enumerate(payloads):
                    # Simple template filling
                    attack_input = template.replace("{payload}", payload)
                    attack_input = attack_input.replace("{benign_query}", "What is 2+2?")
                    attack_input = attack_input.replace("{benign_task}", "help me with math")
                    attack_input = attack_input.replace("{benign}", "Hello")
                    
                    inputs.append({
                        "input": attack_input,
                        "category": category,
                        "attack_id": attack["id"],
                        "attack_name": attack["name"],
                        "severity": attack.get("severity", "medium"),
                        "is_attack": True,
                        "variant_id": i
                    })
        
        return inputs


class PromptLoader:
    """Load system prompts and benign inputs"""
    
    def __init__(self, base_path: str = "data/prompts"):
        self.base_path = Path(base_path)
    
    def load_prompts(self) -> Dict[str, Any]:
        """Load system prompts configuration"""
        prompt_file = self.base_path / "system_prompts.json"
        with open(prompt_file, 'r') as f:
            return json.load(f)
    
    def get_system_prompts(self) -> List[Dict]:
        """Get list of system prompts"""
        data = self.load_prompts()
        return data["evaluation_prompts"]
    
    def get_benign_inputs(self) -> List[str]:
        """Get list of benign inputs"""
        data = self.load_prompts()
        return data["benign_inputs"]


class ExperimentRunner:
    """
    Main experiment runner.
    
    Orchestrates evaluation across models and defense configurations.
    """
    
    def __init__(
        self,
        config: ExperimentConfig,
        model_func_factory: Callable[[str], Callable]
    ):
        """
        Initialize experiment runner.
        
        Args:
            config: Experiment configuration
            model_func_factory: Factory to create model functions
        """
        self.config = config
        self.model_factory = model_func_factory
        
        # Initialize loaders
        self.attack_loader = AttackLoader(config.attack_data_path)
        self.prompt_loader = PromptLoader(config.prompt_data_path)
        
        # Initialize evaluator
        self.evaluator = Evaluator(semantic_threshold=config.semantic_threshold)
        
        # Results storage
        self.results: Dict[str, Any] = {}
    
    def run_full_evaluation(self) -> Dict[str, EvaluationMetrics]:
        """
        Run complete evaluation across all configurations.
        
        Returns:
            Dictionary mapping config names to metrics
        """
        # Load data
        attacks = self.attack_loader.load_all_attacks()
        attack_inputs = self.attack_loader.generate_attack_inputs(attacks)
        system_prompts = self.prompt_loader.get_system_prompts()
        benign_inputs = self.prompt_loader.get_benign_inputs()
        
        results = {}
        
        for model_name in tqdm(self.config.models, desc="Models"):
            model_func = self.model_factory(model_name)
            
            for defense_mode in tqdm(self.config.defense_modes, desc="Defenses", leave=False):
                config_name = f"{model_name}_{defense_mode}"
                
                # Create defense
                defense = DefenseFactory.create(
                    defense_mode,
                    spb_threshold=self.config.spb_threshold,
                    ara_thresholds=self.config.ara_thresholds,
                    rtes_entropy_bits=self.config.rtes_entropy_bits
                )
                
                # Run evaluation
                metrics = self._evaluate_configuration(
                    model_func=model_func,
                    defense=defense,
                    attack_inputs=attack_inputs,
                    benign_inputs=benign_inputs,
                    system_prompts=system_prompts
                )
                
                results[config_name] = metrics
                self.evaluator.reset()
        
        self.results = results
        return results
    
    def _evaluate_configuration(
        self,
        model_func: Callable,
        defense: CombinedDefense,
        attack_inputs: List[Dict],
        benign_inputs: List[str],
        system_prompts: List[Dict]
    ) -> EvaluationMetrics:
        """
        Evaluate a single model-defense configuration.
        
        Args:
            model_func: LLM function
            defense: Defense configuration
            attack_inputs: List of attack inputs
            benign_inputs: List of benign inputs
            system_prompts: List of system prompts
        
        Returns:
            EvaluationMetrics for this configuration
        """
        # Select subset of inputs
        n_attacks = min(len(attack_inputs), self.config.num_prompts_per_config // 2)
        n_benign = self.config.num_prompts_per_config - n_attacks
        
        selected_attacks = np.random.choice(
            len(attack_inputs), n_attacks, replace=False
        )
        
        # Use first system prompt for consistency
        system_prompt = system_prompts[0]["prompt"]
        benign_ref = "Hello, how can I help you today?"
        
        # Get reference output
        reference_output = model_func(system_prompt, benign_ref)
        
        # Evaluate attacks
        for idx in tqdm(selected_attacks, desc="Attacks", leave=False):
            attack = attack_inputs[idx]
            
            start_time = time.time()
            result = defense.defend(
                system_prompt=system_prompt,
                user_input=attack["input"],
                model_func=model_func,
                benign_input=benign_ref
            )
            processing_time = (time.time() - start_time) * 1000
            
            self.evaluator.evaluate_single(
                output=result.final_output,
                reference=reference_output,
                attack_category=attack["category"],
                is_attack=True,
                processing_time_ms=processing_time,
                sre_satisfied=result.sre_satisfied,
                ise_satisfied=result.ise_satisfied,
                cre_satisfied=result.cre_satisfied
            )
        
        # Evaluate benign inputs
        for i in range(min(n_benign, len(benign_inputs))):
            benign = benign_inputs[i % len(benign_inputs)]
            
            start_time = time.time()
            result = defense.defend(
                system_prompt=system_prompt,
                user_input=benign,
                model_func=model_func,
                benign_input=benign_ref
            )
            processing_time = (time.time() - start_time) * 1000
            
            self.evaluator.evaluate_single(
                output=result.final_output,
                reference=reference_output,
                attack_category="benign",
                is_attack=False,
                processing_time_ms=processing_time,
                sre_satisfied=result.sre_satisfied,
                ise_satisfied=result.ise_satisfied,
                cre_satisfied=result.cre_satisfied
            )
        
        return self.evaluator.compute_metrics()
    
    def run_ablation_study(self) -> pd.DataFrame:
        """
        Run ablation study (Table VII).
        
        Returns:
            DataFrame with ablation results
        """
        ablation_configs = [
            ("none", DefenseMode.NONE),
            ("spb_only", DefenseMode.SPB_ONLY),
            ("spb_ara", DefenseMode.SPB_ARA),
            ("spb_ara_rtes", DefenseMode.SPB_ARA_RTES)
        ]
        
        results = []
        
        for config_name, mode in ablation_configs:
            defense = CombinedDefense(mode=mode)
            # Run evaluation...
            # (Implementation similar to run_full_evaluation)
            
            results.append({
                "configuration": config_name,
                "components": self._get_component_string(mode),
                # Metrics would be filled from actual evaluation
            })
        
        return pd.DataFrame(results)
    
    def _get_component_string(self, mode: DefenseMode) -> str:
        """Get component string for ablation table"""
        components = {
            DefenseMode.NONE: "—",
            DefenseMode.SPB_ONLY: "SPB",
            DefenseMode.SPB_ARA: "SPB+ARA",
            DefenseMode.SPB_ARA_RTES: "SPB+ARA+RTES"
        }
        return components.get(mode, "Unknown")
    
    def run_fingerprinting_evaluation(self) -> Dict[str, Dict[str, float]]:
        """
        Run fingerprinting resistance evaluation (Table IX).
        
        Returns:
            Dictionary with fingerprinting results per defense config
        """
        fingerprint_eval = FingerprintingEvaluator(n_probes=8)
        results = {}
        
        configs = [
            ("spb_only_deterministic", DefenseMode.SPB_ONLY),
            ("spb_ara_deterministic", DefenseMode.SPB_ARA),
            ("spb_ara_rtes", DefenseMode.SPB_ARA_RTES)
        ]
        
        for config_name, mode in configs:
            defense = CombinedDefense(mode=mode)
            
            def defense_func(inp):
                return defense.defend(
                    system_prompt="You are helpful.",
                    user_input=inp,
                    model_func=lambda s, x: f"Response to: {x}",
                    benign_input="Hello"
                )
            
            metrics = fingerprint_eval.evaluate_fingerprinting(defense_func)
            results[config_name] = metrics
        
        return results
    
    def save_results(self, output_dir: Optional[str] = None):
        """Save all results to files"""
        output_path = Path(output_dir or self.config.output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save main results
        if self.results:
            results_dict = {}
            for name, metrics in self.results.items():
                results_dict[name] = asdict(metrics)
            
            with open(output_path / "evaluation_results.json", 'w') as f:
                json.dump(results_dict, f, indent=2, default=str)
    
    def generate_latex_tables(self) -> Dict[str, str]:
        """Generate LaTeX tables for paper"""
        tables = {}
        
        # Table V: Main results
        if self.results:
            table_v = self._generate_table_v()
            tables["table_v"] = table_v
        
        return tables
    
    def _generate_table_v(self) -> str:
        """Generate Table V (Main Results) in LaTeX format"""
        latex = r"""
\begin{table*}[!t]
\renewcommand{\arraystretch}{1.3}
\caption{Main Evaluation Results}
\label{tab:main_results}
\centering
\begin{tabular}{llcccc}
\toprule
\textbf{Model} & \textbf{Defense} & \textbf{ASR (\%)} & \textbf{SF (\%)} & \textbf{Overhead (ms)} & \textbf{FPR (\%)} \\
\midrule
"""
        
        for config_name, metrics in self.results.items():
            model, defense = config_name.rsplit('_', 1)
            latex += f"{model} & {defense} & {metrics.asr*100:.1f} & {metrics.semantic_fidelity*100:.1f} & {metrics.overhead_ms:.1f} & {metrics.fpr*100:.1f} \\\\\n"
        
        latex += r"""
\bottomrule
\end{tabular}
\end{table*}
"""
        return latex


def load_config(config_path: str) -> ExperimentConfig:
    """Load experiment configuration from YAML file"""
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    return ExperimentConfig(**config_dict)
