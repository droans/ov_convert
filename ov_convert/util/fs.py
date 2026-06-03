"""Utility functions related to the filesystem."""

import json
from pathlib import Path

import yaml

from ov_convert.models import LlmModelConfiguration, VlmModelConfiguration
from ov_convert.models.model import ModelConfigurationInternal
from ov_convert.util.log import logger


def get_config_from_file(config_file_path: str) -> ModelConfigurationInternal:
    """Load the data from `config_file_path` into the ModelConfigurationInternal model."""
    with open(config_file_path) as f:
        model = ModelConfigurationInternal(config=yaml.safe_load(f.read()))
    if not model.config.export.path:
        model.config.export.path = Path(config_file_path).parent
    return model


def dump_config_to_file(config: VlmModelConfiguration | LlmModelConfiguration) -> None:
    """Dump the configuration to a file."""
    include_defaults = config.export.include_defaults
    msg = f"Include defaults: {include_defaults}"
    logger.error(msg)

    # Must dump to json and reload due to PosixPath objects in schema
    data = json.loads(config.model_dump_json(exclude_unset=not include_defaults))

    yaml_str = yaml.safe_dump(data)
    save_path = f"{config.export.path}/conversion_config.yaml"
    with open(save_path, "w") as f:
        f.write(yaml_str)


def create_directory_if_nonexistant(directory: str | Path) -> None:
    """Create a directory if it doesn't exist. Raises an error if the parent does not exist."""
    if isinstance(directory, str):
        directory = Path(directory)
    if directory.exists():
        return
    if not directory.parent.exists():
        msg = f"Cannot create path at {directory}. Does the parent directory exist?"
        raise FileNotFoundError(msg)
    Path.mkdir(directory)
