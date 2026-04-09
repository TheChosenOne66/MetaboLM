"""Configuration system using dataclasses + YAML."""

from dataclasses import dataclass, field
from pathlib import Path
import yaml


@dataclass
class DataConfig:
    train_path: str = "data/processed/train.csv"
    val_path: str = "data/processed/val.csv"
    correlation_matrix_path: str = "data/processed/correlation_matrix.pt"
    num_metabolites: int = 168
    # Raw data paths (for data preparation)
    raw_metabolomics_path: str = ""
    raw_diagnoses_path: str = ""
    raw_first_occur_path: str = ""
    raw_cause_of_death_path: str = ""
    raw_time_blood_path: str = ""


@dataclass
class ModelConfig:
    pretrained_ckpt: str = "weights/best_metabolite_bert_model.pt"
    head_type: str = "single_task"  # single_task | hierarchical
    freeze_strategy: str = "none"   # none | head_only | adapter | lora
    hidden_size: int = 768
    num_diseases: int = 16
    proj_size: int = 256            # For hierarchical head shared projection
    adapter_bottleneck: int = 64
    lora_rank: int = 8
    sft_ckpt: str = ""              # For RL: path to best SFT checkpoint


@dataclass
class TrainingConfig:
    mode: str = "sft"               # sft | rl
    batch_size: int = 512
    learning_rate: float = 2e-5
    num_epochs: int = 40
    early_stopping_patience: int = 10
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    # Hierarchical loss weights
    lambda_chapter: float = 1.0
    mu_hierarchy: float = 0.1
    # RL-specific (GRPO)
    reward: str = "calibration"     # calibration | hierarchy
    group_size: int = 8
    kl_coeff: float = 0.1


@dataclass
class ExperimentConfig:
    experiment: str = "default"
    seed: int = 42
    output_dir: str = "outputs/default"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


def load_config(path: str) -> ExperimentConfig:
    """Load YAML config file and return ExperimentConfig.

    Supports nested keys matching dataclass structure:
        data:
          train_path: ...
        model:
          head_type: ...
        training:
          mode: ...
    """
    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    data_cfg = DataConfig(**{k: v for k, v in raw.get("data", {}).items() if hasattr(DataConfig, k)})
    model_cfg = ModelConfig(**{k: v for k, v in raw.get("model", {}).items() if hasattr(ModelConfig, k)})
    training_cfg = TrainingConfig(**{k: v for k, v in raw.get("training", {}).items() if hasattr(TrainingConfig, k)})

    return ExperimentConfig(
        experiment=raw.get("experiment", "default"),
        seed=raw.get("seed", 42),
        output_dir=raw.get("output_dir", "outputs/default"),
        data=data_cfg,
        model=model_cfg,
        training=training_cfg,
    )
