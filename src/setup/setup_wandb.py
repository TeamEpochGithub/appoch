from omegaconf import DictConfig
import wandb
from src.utils.logger import logger
from pathlib import Path


def setup_wandb(
    cfg: DictConfig,
    job_type: str,
    output_dir: Path,
    name: str | None = None,
    group: str | None = None,
) -> wandb.sdk.wandb_run.Run | wandb.sdk.lib.RunDisabled | None:
    """Initialize Weights & Biases and log the config and code.

    :param cfg: The config object. Created with Hydra or OmegaConf.
    :param job_type: The type of job, e.g. Training, CV, etc.
    :param output_dir: The directory to the Hydra outputs.
    :param name: The name of the run.
    :param group: The namer of the group of the run.
    """
    logger.debug("Initializing Weights & Biases")

    config = OmegaConf.to_container(cfg, resolve=True)
    run = wandb.init(
        config=replace_list_with_dict(config),  # type: ignore[arg-type]
        project="detect-harmful-brain-activity",
        entity="team-epoch-iv",
        name=name,
        group=group,
        job_type=job_type,
        tags=cfg.wandb.tags,
        notes=cfg.wandb.notes,
        settings=wandb.Settings(start_method="thread", code_dir="."),
        dir=output_dir,
        reinit=True,
    )

    if isinstance(run, wandb.sdk.lib.RunDisabled) or run is None:  # Can't be True after wandb.init, but this casts wandb.run to be non-None, which is necessary for MyPy
        raise RuntimeError("Failed to initialize Weights & Biases")

    if cfg.wandb.log_config:
        logger.debug("Uploading config files to Weights & Biases")

        # Get the config file name
        if job_type == "sweep":
            job_type = "cv"
        curr_config = "conf/" + job_type + ".yaml"

        # Get the model file name
        if "model" in cfg:
            model_name = OmegaConf.load(curr_config).defaults[2].model
            model_path = f"conf/model/{model_name}.yaml"
        elif "ensemble" in cfg:
            model_name = OmegaConf.load(curr_config).defaults[2].ensemble
            model_path = f"conf/ensemble/{model_name}.yaml"
        elif "post_ensemble" in cfg:
            model_name = OmegaConf.load(curr_config).defaults[2].post_ensemble
            model_path = f"conf/post_ensemble/{model_name}.yaml"

        # Store the config as an artefact of W&B
        artifact = wandb.Artifact(job_type + "_config", type="config")
        config_path = output_dir / ".hydra/config.yaml"
        artifact.add_file(str(config_path), "config.yaml")
        artifact.add_file(curr_config)
        artifact.add_file(model_path)
        wandb.log_artifact(artifact)

    if cfg.wandb.log_code.enabled:
        logger.debug("Uploading code files to Weights & Biases")

        run.log_code(
            root=".",
            exclude_fn=cast(Callable[[str, str], bool], lambda abs_path, root: re.match(cfg.wandb.log_code.exclude, Path(abs_path).relative_to(root).as_posix()) is not None),
        )

    logger.info("Done initializing Weights & Biases")
    return run