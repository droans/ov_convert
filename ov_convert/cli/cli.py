"""CLI Utility for ov-convert."""

import argparse
import sys
from pathlib import Path

import yaml


def cli_convert() -> None:
    """CLI conversion function."""
    parser = argparse.ArgumentParser(
        description="""ov-convert - OpenVINO Conversion Utility

  Convert a VLM or LLM to an OpenVINO compatible model using a YAML configuration file.
  """,
    )
    parser.add_argument(
        "config_file",
        action="store",
        help="Path to config file.",
        nargs="?",
    )
    args = parser.parse_args()
    if not args.config_file:
        parser.print_help()
        sys.exit()
    file_path = args.config_file
    if not test_config_file(file_path):
        print(  # noqa: T201
            f"`{file_path}` either doesn't exist, is not a YAML file,"
            "and/or contains invalid configuration.",
        )
        sys.exit()
    from ov_convert.convert import export_from_config_file

    export_from_config_file(file_path)


def test_config_file(file_path: str) -> bool:
    """Tests if config file path exists and if config file loads properly."""
    from ov_convert.models.model import ModelConfigurationInternal

    if not Path.exists(Path(file_path)) or not file_path.endswith((".yaml", ".yml")):
        return False
    with open(file_path) as f:
        data = yaml.safe_load(f.read())
    ModelConfigurationInternal(config=data)
    return True
