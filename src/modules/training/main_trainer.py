"""Module for example training block."""

from dataclasses import dataclass

import wandb
from epochalyst.pipeline.model.training.torch_trainer import TorchTrainer

from src.modules.logging.logger import Logger


@dataclass
class MainTrainer(TorchTrainer, Logger):
    """Main training block."""

    def save_model_to_external(self) -> None:
        """Save the model to external storage."""
        if wandb.run:
            model_artifact = wandb.Artifact(self.model_name, type="model")
            model_artifact.add_file(f"{self._model_directory}/{self.get_hash()}.pt")
            wandb.log_artifact(model_artifact)
