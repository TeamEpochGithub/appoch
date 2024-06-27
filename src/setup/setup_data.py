"""File containing all methods related to processing and formatting data before it is inserted into a pipeline.

- Since these methods are very competition specific none have been implemented here yet.
- Usually you'll have one data setup for training and another for making submissions.
"""

from os import PathLike
from typing import Any


def setup_train_x_data(raw_path: PathLike[str]) -> Any:  # noqa: ARG001, ANN401
    """Create train x data for pipeline.

    :param raw_path: Path to the raw data.
    :raises NotImplementedError: Implement this function per competition.
    """
    raise NotImplementedError("Setup train data X is competition specific. Implement within competition repository")


def setup_train_y_data(raw_path: PathLike[str]) -> Any:  # noqa: ARG001, ANN401
    """Create train y data for pipeline.

    :param raw_path: Path to the raw data.
    :raises NotImplementedError: Implement this function per competition.
    """
    raise NotImplementedError("Setup train data y is competition specific. Implement within competition repository")


def setup_inference_data(raw_path: PathLike[str]) -> Any:  # noqa: ARG001, ANN401
    """Create data for inference with pipeline.

    :param raw_path: Path to the raw data.
    :raises NotImplementedError: Implement this function per competition.
    """
    raise NotImplementedError("Setup inference data is competition specific. Implement within competition repository. It might be the same as setup_train_x_data.")


def setup_splitter_data() -> Any:  # noqa: ANN401
    """Create data for splitter.

    :raises NotImplementedError: Implement this function per competition.
    """
    raise NotImplementedError("Setup splitter data is competition specific. Implement within competition repository.")
