"""CLI Argument Model."""

from typing import Literal

from pydantic import AliasChoices, BaseModel, Field
from pydantic_settings import (
    BaseSettings,
    CliPositionalArg,
    CliSubCommand,
)

from ov_convert.cli.const import (
    AptDependencyGroupsType,
    PipDependencyGroupOptionsType,
    PipDependencyOptionsType,
)


class ConvertModelSchema(BaseModel):
    """Convert OV model: Config schema."""

    config_path: CliPositionalArg[str]


class DependencyManagementModelSchema(BaseModel):
    """Dependency Management: Install/update dependencies."""

    action: CliPositionalArg[Literal["install", "upgrade"]]
    dependency: CliPositionalArg[
        Literal["all"]
        | AptDependencyGroupsType
        | PipDependencyOptionsType
        | PipDependencyGroupOptionsType
    ]
    branch: str | None = Field(
        default=None,
        alias=AliasChoices("b", "branch"),
    )  # ty:ignore[no-matching-overload]
    pr: int | None = Field(
        default=None,
        alias=AliasChoices("pr", "pull-request"),
    )  # ty:ignore[no-matching-overload]


class CLIModelSchema(BaseSettings, cli_parse_args=True, cli_exit_on_error=False):
    """Model for CLI arguments."""

    convert: CliSubCommand[ConvertModelSchema]
