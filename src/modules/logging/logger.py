"""A logger that logs to the terminal and to W&B."""

from typing import Any

import wandb
from epochalyst._core._logging._logger import _Logger

from src.utils.logger import logger


class Logger(_Logger):
    """A logger that logs to the terminal and to W&B.

    To use this logger, inherit, this will make the following methods available:
    - log_to_terminal(message: str) -> None
    - log_to_debug(message: str) -> None
    - log_to_warning(message: str) -> None
    - log_to_external(message: dict[str, Any], **kwargs: Any) -> None
    - external_define_metric(metric: str, metric_type: str) -> None
    """

    def log_to_terminal(self, message: str) -> None:
        """Log a message to the terminal.

        :param message: The message to log
        """
        logger.info(message)

    def log_to_debug(self, message: str) -> None:
        """Log a message to the debug level.

        :param message: The message to log
        """
        logger.debug(message)

    def log_to_warning(self, message: str) -> None:
        """Log a message to the warning level.

        :param message: The message to log
        """
        logger.warning(message)

    def log_to_external(self, message: dict[str, Any], **kwargs: Any) -> None:
        """Log a message to an external service.

        :param message: The message to log
        :param kwargs: Any additional arguments
        """
        if wandb.run:
            if message.get("type") == "wandb_plot" and message["plot_type"] == "line_series":
                plot_data = message["data"]
                # Construct the plot here using the provided data
                plot = wandb.plot.line_series(
                    xs=plot_data["xs"],
                    ys=plot_data["ys"],
                    keys=plot_data["keys"],
                    title=plot_data["title"],
                    xname=plot_data["xname"],
                )
                wandb.log({plot_data["title"]: plot}, commit=False, **kwargs)
            else:
                wandb.log(message, **kwargs)

    def external_define_metric(self, metric: str, metric_type: str) -> None:
        """Define a metric in an external service.

        :param metric: The metric to define
        :param metric_type: The type of the metric
        """
        if wandb.run:
            wandb.define_metric(metric, step_metric=metric_type)
