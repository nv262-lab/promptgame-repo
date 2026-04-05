# PromptGame: Game-Theoretic Defense Against Prompt Injection Attacks

Implementation of the PromptGame framework for semantic integrity preservation in LLM systems under adversarial prompt injection attacks.

## Repository Structure

```
promptgame/
├── src/
│   ├── defenses/          # Defense mechanisms (SPB, ARA, RTES)
│   ├── attacks/           # Attack implementations
│   ├── evaluation/        # Evaluation metrics and runners
│   ├── framework/         # PromptGame framework core
│   └── utils/             # Utility functions
├── configs/               # Configuration files
├── data/
│   ├── attacks/           # Attack datasets (50 variants)
│   ├── prompts/           # System prompts and benign inputs
│   └── results/           # Evaluation results
├── scripts/               # Execution scripts
└── tests/                 # Unit tests
```

## Installation

```bash
# Clone repository
git clone https://github.com/[username]/promptgame.git
cd promptgame

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## Requirements

- Python 3.9+
- PyTorch 2.0+
- transformers 4.35+
- openai 1.0+
- anthropic 0.18+
- bert-score 0.3.13+
- numpy, pandas, scipy

## Quick Start

### Run Full Evaluation

```bash
# Reproduce main results (Table V)
python scripts/run_evaluation.py --config configs/main_evaluation.yaml

# Run ablation study (Table VII)
python scripts/run_ablation.py --config configs/ablation.yaml

# Run fingerprinting experiment (Table IX)
python scripts/run_fingerprinting.py --config configs/fingerprinting.yaml
```

### Individual Defense Testing

```bash
# Test SPB defense
python -m src.defenses.spb --input "test prompt" --system-prompt "You are a helpful assistant."

# Test combined defense (SPB + ARA + RTES)
python -m src.defenses.combined --input "test prompt" --config configs/defense_config.yaml
```

## Defense Mechanisms

### 1. Semantic Prompt Binding (SPB)
Enforces semantic consistency between defended outputs and reference outputs.

```python
from src.defenses import SemanticPromptBinding

spb = SemanticPromptBinding(threshold=0.85)
result = spb.defend(system_prompt, user_input, model)
```

### 2. Adaptive Risk Assessment (ARA)
Dynamically adjusts defense intensity based on input risk scoring.

```python
from src.defenses import AdaptiveRiskAssessment

ara = AdaptiveRiskAssessment(risk_thresholds=[0.3, 0.6, 0.9])
risk_level, action = ara.assess(user_input)
```

### 3. Real-Time Equilibrium Strategy (RTES)
Implements a randomized mixed-strategy defense that continuously updates defense configurations based on observed attack patterns using multiplicative weights, preventing adversarial fingerprinting of the defense mechanism.

```python
from src.defenses import RealTimeEquilibriumStrategy

rtes = RealTimeEquilibriumStrategy(k=8, eta=0.1, window=100)
defended_input = rtes.apply(user_input)
```

## Attack Dataset

50 attack variants across 4 categories:

| Category | Count | Source |
|----------|-------|--------|
| Direct Injection | 15 | TensorTrust |
| RAG Poisoning | 12 | PoisonedRAG |
| Separator/Delimiter | 13 | Li et al. |
| Cascading Agent | 10 | InjecAgent |

## Evaluation Metrics

- **ASR (Attack Success Rate)**: Fraction of attacks achieving σ < τ
- **SF (Semantic Fidelity)**: Average semantic similarity score
- **Overhead**: Additional processing time (ms)
- **FPR (False Positive Rate)**: Benign inputs incorrectly flagged

## Configuration

Edit `configs/main_evaluation.yaml`:

```yaml
models:
  - gpt-4o
  - claude-3-sonnet
  - llama-3-70b
  - mistral-7b

defense:
  spb_threshold: 0.85
  ara_risk_thresholds: [0.3, 0.7]
  rtes_k: 8
  rtes_eta: 0.1
  rtes_window: 100

evaluation:
  num_prompts: 2500
  num_runs: 5
  semantic_threshold: 0.85
```

## API Keys

Set environment variables:

```bash
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
export HF_TOKEN="your-huggingface-token"
```

## Reproducing Paper Results

```bash
# Table V: Main results
python scripts/run_evaluation.py --output results/table_v.csv

# Table VI: Per-category results
python scripts/run_per_category.py --output results/table_vi.csv

# Table VII: Ablation study
python scripts/run_ablation.py --output results/table_vii.csv

# Table VIII: Equilibrium convergence
python scripts/run_convergence.py --output results/table_viii.csv

# Table IX: Fingerprinting resistance
python scripts/run_fingerprinting.py --output results/table_ix.csv

# Table X-XI: LLM-as-judge validation
python scripts/run_judge_validation.py --output results/table_x_xi.csv
```

## Citation

```bibtex
@article{vummaneni2026semantic,
  title={Semantic Integrity Under Prompt Injection: A Game-Theoretic Analysis of Equilibria in Adversarial Natural Language Processing Systems},
  author={Vummaneni, Naga Sujitha and Bobba, Sundeep and Mittal, Adarsh and Kumar, Ishan and Jammula, Usha Ratnam},
  journal={IEEE Transactions on Information Forensics and Security},
  year={2026}
}
```

## License

MIT License
