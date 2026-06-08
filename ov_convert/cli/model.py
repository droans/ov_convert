"""CLI Argument Model."""

from pydantic import BaseModel
from pydantic_settings import (
    BaseSettings,
    CliPositionalArg,
    CliSubCommand,
)


class ConvertModelSchema(BaseModel):
    """Convert OV model: Config schema."""

    config_path: CliPositionalArg[str]


class CLIModelSchema(BaseSettings, cli_parse_args=True, cli_exit_on_error=False):
    """Model for CLI arguments."""

    convert: CliSubCommand[ConvertModelSchema]
