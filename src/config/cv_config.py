"""Schema for the cross validation configuration."""

from dataclasses import dataclass
from typing import Any

from src.config.wandb_config import WandBConfig


@dataclass
class CVConfig:
    """Schema for the cross validation configuration.

    :param model: Model pipeline.
    :param ensemble: Ensemble pipeline.
    :param raw_data_path: Path to the raw data.
    :param raw_target_path: Path to the raw target.
    :param processed_path: Path to put processed data.
    :param scorer: Scorer object to be instantiated.
    :param wandb: Whether to log to Weights & Biases and other settings.
    :param splitter: Cross validation splitter.
    :param allow_multiple_instances: Whether to allow multiple instances of training at the same time.
    :param save_folds: Whether to save the fold models
    """

    model: Any
    ensemble: Any
    raw_path: str
    cache_path: str
    data_path: str
    processed_path: str
    scorer: Any
    wandb: WandBConfig
    splitter: Any
    allow_multiple_instances: bool = False
    save_folds: bool = True
