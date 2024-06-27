"""A verbose training pipeline that logs to the terminal and to W&B."""

from epochalyst.pipeline.model.training.training import TrainingPipeline

from src.modules.logging.logger import Logger


class VerboseTrainingPipeline(TrainingPipeline, Logger):
    """A verbose training pipeline that logs to the terminal and to W&B."""
