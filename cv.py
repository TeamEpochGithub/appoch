"""The main script for Cross Validation. Takes in the raw data, does CV and logs the results."""

import os
import warnings
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import randomname
import wandb
from epochalyst.logging.section_separator import print_section_separator
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
from omegaconf import DictConfig

from src.config.cv_config import CVConfig
from src.scoring.scorer import Scorer
from src.setup.setup_data import setup_train_x_data, setup_train_y_data
from src.setup.setup_pipeline import setup_pipeline
from src.setup.setup_runtime_args import setup_train_args
from src.setup.setup_wandb import setup_wandb
from src.utils.lock import Lock
from src.utils.logger import logger
from src.utils.set_torch_seed import set_torch_seed

warnings.filterwarnings("ignore", category=UserWarning)
# Makes hydra give full error messages
os.environ["HYDRA_FULL_ERROR"] = "1"
# Set up the config store, necessary for type checking of config yaml
cs = ConfigStore.instance()
cs.store(name="base_cv", node=CVConfig)


@hydra.main(version_base=None, config_path="conf", config_name="cv")
def run_cv(cfg: DictConfig) -> None:  # TODO(Jeffrey): Use CVConfig instead of DictConfig
    """Do cv on a model pipeline with K fold split. Entry point for Hydra which loads the config file."""
    # Run the cv config with an optional lock
    optional_lock = Lock if not cfg.allow_multiple_instances else nullcontext
    with optional_lock():
        run_cv_cfg(cfg)


def run_cv_cfg(cfg: DictConfig) -> None:
    """Do cv on a model pipeline with K fold split."""
    print_section_separator("Q? - 'competition' - CV")

    import coloredlogs

    coloredlogs.install()

    # Set seed
    set_torch_seed()

    # Get output directory
    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)

    # Set up Weights & Biases group name
    if cfg.wandb.enabled:
        wandb_group_name = randomname.get_name()
        setup_wandb(cfg, "cv", output_dir, name=wandb_group_name, group=wandb_group_name)

    # Preload the pipeline
    print_section_separator("Setup pipeline")
    model_pipeline = setup_pipeline(cfg)

    # Cache arguments for x_sys
    processed_data_path = Path(cfg.processed_path)
    processed_data_path.mkdir(parents=True, exist_ok=True)
    cache_args = {
        "output_data_type": "numpy_array",
        "storage_type": ".pkl",
        "storage_path": f"{processed_data_path}",
    }

    # Read the data if required and split in X, y
    x_cache_exists = model_pipeline.get_x_cache_exists(cache_args)
    y_cache_exists = model_pipeline.get_y_cache_exists(cache_args)

    X, y = None, None
    if not x_cache_exists:
        X = setup_train_x_data(cfg.data_path)

    if not y_cache_exists:
        y = setup_train_y_data(cfg.data_path)

    # Instantiate scorer
    scorer = instantiate(cfg.scorer)
    scores: list[float] = []

    # Split indices into train and test
    # splitter_data = setup_splitter_data()
    logger.info("Using splitter to split data into train and test sets.")

    if not isinstance(y, np.ndarray):
        raise TypeError("y should be a numpy array")

    oof_predictions = np.zeros(y.shape, dtype=np.float64)

    for fold_no, (train_indices, test_indices) in enumerate(instantiate(cfg.splitter).split(y)):
        score, predictions = run_fold(fold_no, X, y, train_indices, test_indices, cfg, scorer, output_dir, cache_args)
        scores.append(score)

        # Save predictions
        oof_predictions[test_indices] = predictions

    avg_score = np.average(np.array(scores))
    oof_score = scorer(y, oof_predictions)

    print_section_separator("CV - Results")
    logger.info(f"Avg Score: {avg_score}")
    wandb.log({"Avg Score": avg_score})

    logger.info(f"OOF Score: {oof_score}")
    wandb.log({"OOF Score": oof_score})

    logger.info("Finishing wandb run")
    wandb.finish()


def run_fold(
    fold_no: int,
    X: Any,  # noqa: ANN401
    y: Any,  # noqa: ANN401
    train_indices: list[int],
    test_indices: list[int],
    cfg: DictConfig,
    scorer: Scorer,
    output_dir: Path,
    cache_args: dict[str, Any],
) -> tuple[float, Any]:
    """Run a single fold of the cross validation.

    :param i: The fold number.
    :param X: The input data.
    :param y: The labels.
    :param train_indices: The indices of the training data.
    :param test_indices: The indices of the test data.
    :param cfg: The config file.
    :param scorer: The scorer to use.
    :param output_dir: The output directory for the prediction plots.
    :param processed_y: The processed labels.
    :return: The score of the fold and the predictions.
    """
    # Print section separator
    print_section_separator(f"CV - Fold {fold_no}")
    logger.info(f"Train/Test size: {len(train_indices)}/{len(test_indices)}")

    logger.info("Creating clean pipeline for this fold")
    model_pipeline = setup_pipeline(cfg)

    train_args = setup_train_args(
        pipeline=model_pipeline,
        cache_args=cache_args,
        train_indices=train_indices,
        test_indices=test_indices,
        fold=fold_no,
        save_model=cfg.save_folds,
    )
    predictions, _ = model_pipeline.train(X, y, **train_args)

    # If predictions are on 'all' data, only keep the test data
    if y.shape[0] == predictions.shape[0]:
        predictions = predictions[test_indices]
    score = scorer(y[test_indices], predictions)

    logger.info(f"Score, fold {fold_no}: {score}")

    fold_dir = output_dir / str(fold_no)  # Files specific to a run can be saved here
    logger.debug(f"Output Directory: {fold_dir}")

    if wandb.run:
        wandb.log({f"Score_{fold_no}": score})
    return score, predictions


if __name__ == "__main__":
    run_cv()
