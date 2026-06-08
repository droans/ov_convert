"""Conversion CLI functions."""

from enum import Enum
from pathlib import Path

import yaml

from ov_convert.cli.model import ConvertModelSchema


class ConfigFileError(Enum):
    """Enum class for possible config file errors."""

    OK = 1
    DOES_NOT_EXIST = -1
    NOT_VALID = -2
    BAD_EXTENSTION = -3


def cli_convert(config: ConvertModelSchema) -> None:
    """Convert a model via CLI."""
    file_path = config.config_path
    config_file_result = test_config_file(file_path)
    if config_file_result != ConfigFileError.OK:
        if config_file_result == ConfigFileError.BAD_EXTENSTION:
            print(f"Extension for `{file_path}` is invalid: Must be either `.yaml` or `.yml`")
        elif config_file_result == ConfigFileError.DOES_NOT_EXIST:
            print(f"Could not find any file at `{file_path}`!")
        elif config_file_result == ConfigFileError.NOT_VALID:
            print(f"Could not validate file `{file_path}`. Please check any errors above.")
        else:
            print("Received unknown error!")
        return
    from ov_convert.convert import export_from_config_file

    export_from_config_file(file_path)


def test_config_file(file_path: str) -> ConfigFileError:
    """Tests if config file path exists and if config file loads properly."""
    from ov_convert.models.model import ModelConfigurationInternal

    path = Path(file_path)
    if not path.exists():
        return ConfigFileError.DOES_NOT_EXIST
    if not file_path.endswith((".yaml", ".yml")):
        return ConfigFileError.BAD_EXTENSTION
    with open(file_path) as f:
        data = yaml.safe_load(f.read())
    ModelConfigurationInternal(config=data)
    return ConfigFileError.OK
