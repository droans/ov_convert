"""CLI Utility for ov-convert."""

import sys
from collections.abc import Callable

from pydantic import BaseModel
from pydantic_settings import get_subcommand

from ov_convert.cli.deps import cli_manage_dependencies
from ov_convert.cli.model import CLIModelSchema, DependencyManagementModelSchema

SUBCOMMAND_FUNCS: dict[type[BaseModel], Callable] = {
    DependencyManagementModelSchema: cli_manage_dependencies,
}


def cli_convert() -> None:
    """CLI conversion function."""
    args = sys.argv
    if "-h" in args or "--help" in args or len(args) < 2:  # noqa: PLR2004
        print("""ov-convert - OpenVINO Conversion Utility

  Convert a VLM or LLM to an OpenVINO compatible model using a YAML configuration file.
        """)  #  noqa: T201
        sys.exit()
    if len(sys.argv) > 2:  # noqa: PLR2004
        subcommand: BaseModel = get_subcommand(CLIModelSchema())  # ty:ignore[missing-argument, invalid-assignment]
        subcommand_func = SUBCOMMAND_FUNCS[type(subcommand)]
    else:
        subcommand = cli_convert
    subcommand_func(subcommand)
