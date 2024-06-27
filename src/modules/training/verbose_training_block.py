"""Module for a verbose training block that logs to the terminal and to W&B."""

from epochalyst.pipeline.model.training.training_block import TrainingBlock

from src.modules.logging.logger import Logger


class VerboseTrainingBlock(TrainingBlock, Logger):
    """A verbose training block that logs to the terminal and to W&B.

    To use this block, inherit and implement the following methods:
    - train(x: Any, y: Any, **kwargs: Any) -> tuple[Any, Any]
    - predict(x: Any, **kwargs: Any) -> Any
    """
