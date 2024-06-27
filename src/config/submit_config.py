"""Schema for the submit configuration."""

from dataclasses import dataclass
from typing import Any


@dataclass
class SubmitConfig:
    """Schema for the submit configuration.

    :param model: Model pipeline.
    :param ensemble: Ensemble pipeline.
    :param raw_data_path: Path to the raw data.
    :param raw_target_path: Path to the raw target.
    :param result_path: Path to the result.
    """

    model: Any
    ensemble: Any
    post_ensemble: Any
    raw_path: str
    cache_path: str
    data_path: str
    processed_path: str
    result_path: str
