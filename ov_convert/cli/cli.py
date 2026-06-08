"""CLI Utility for ov-convert."""

import sys
from collections.abc import Callable

from pydantic_settings import get_subcommand

from ov_convert.cli.convert import cli_convert
from ov_convert.cli.deps import cli_manage_dependencies
from ov_convert.cli.model import (
    CLIModelSchema,
    ConvertModelSchema,
    DependencyManagementModelSchema,
    SubcommandSchemaType,
)
from ov_convert.cli.util import print_help

SUBCOMMAND_FUNCS: dict[type[SubcommandSchemaType], Callable] = {
    DependencyManagementModelSchema: cli_manage_dependencies,
    ConvertModelSchema: cli_convert,
}


def cli_call() -> None:
    """Handle `ov-convert` shell requests."""
    args = sys.argv
    if "-h" in args or "--help" in args or len(args) < 2:  # noqa: PLR2004
        print_help()
    if len(sys.argv) > 2:  # noqa: PLR2004
        subcommand: SubcommandSchemaType = get_subcommand(CLIModelSchema())  # ty:ignore[missing-argument, invalid-assignment]
        subcommand_func = SUBCOMMAND_FUNCS[type(subcommand)]
    else:
        subcommand_func = cli_convert
        subcommand = ConvertModelSchema(config_path=sys.argv[-1])
    subcommand_func(subcommand)  # ty:ignore[invalid-argument-type]
