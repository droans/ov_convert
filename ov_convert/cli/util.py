"""CLI Utility functions."""

import os
import shutil
import subprocess
import sys
from typing import Literal

from ov_convert.cli.const import (
    APT_GROUP_DEPENDENCIES,
    PIP_GROUP_DEPENDENCIES,
    AptDependencyGroupsType,
    PipDependencyGroupOptionsType,
    PipDependencyOptionsType,
)


def print_help() -> None:
    """Print help to stdout."""
    term_width = shutil.get_terminal_size((80, 20)).columns

    def print_centered(line: str) -> None:
        """Print a single centered line."""
        pad_cols = 0 if len(line) >= term_width else round((term_width - len(line)) / 2)
        pad = " " * pad_cols
        print(f"{pad}{line}")

    def print_seperator(sep: str = "-") -> None:
        """Print a seperator."""
        print(sep * term_width)

    print_centered("ov-convert")
    print("""
Description:
    Convert a VLM or LLM to an OpenVINO compatible model using a YAML configuration file.
""")
    print_seperator()
    print("""
Usage:
    Convert a model using a config file:
        ov-convert [convert] config.yaml

    Install/Update OpenVINO dependencies:
        ov-convert dep [upgrade|install] [dependency|dependency-group]
            Dependencies [group(s)]:
                * openvino [all-ov, all-pip, all]
                * openvino-genai [all-ov, all-pip, all]
                * openvino-tokenizers [all-ov, all-pip, all]
                * optimum [all-optimum, all-pip, all]
                * optimum-intel [all-optimum, all-pip, all]
                * optimum-onnx [all-optimum, all-pip, all]
                * transformers [all-pip, all]
                * IGC/ICR/L0 [all-compute, all]
                    NOTE:   Intel Graphics Compiler, Intel Compute Runtime,
                            and Level Zero packages are shared dependencies
                            and cannot be installed individually.
Parameters:
    ov-convert deps:
        When upgrading/installing individual pip dependencies,
        the following parameters are available:
            -b, --branch            Specify the branch used by the dependency
            -pr, --pull-request     Specify a specific pull request to use for
                                    the dependency
""")


def get_bin_path(bin_name: str, error_out: bool = True) -> str:
    """Get the path for a binary using `which {bin}`."""
    cmd_result = subprocess.run(["which", bin_name], check=True, capture_output=True)  # noqa: S603, S607
    if cmd_result.returncode != 0:
        msg = f"Could not get binary for {bin_name} (Return code: {cmd_result.returncode})"
        if error_out:
            raise SystemError(msg)
        print(f"Received error: {msg}")
        return ""
    return cmd_result.stdout.decode().replace("\n", "")


def is_sudo_present() -> bool:
    """Test if `sudo` is present on the machine."""
    path = get_bin_path("sudo")
    return len(path) > 0


def get_pip_env_cmd() -> list[str]:
    """Get the proper pip command based on the presence of the venv."""
    if "OPENARC_API_KEY" in os.environ:
        # OpenArc - use default `uv` executables.
        uv_path = get_bin_path("uv")
        cmd = [
            uv_path,
            "--directory",
            "/app",
        ]
    else:
        cmd = [sys.executable, "-m"]
    return [
        *cmd,
        "pip",
        "install",
    ]


def install_apt_dependency(
    dep: str | list[str],
    install_or_upgrade: Literal["install", "upgrade"],
) -> None:
    """Installs a single or multiple apt dependencies."""
    if isinstance(dep, list):
        dep = " ".join(dep)
    update_cmd: list[str] = []
    cmd: list[str] = []
    if is_sudo_present():
        print("sudo found on machine. You may be asked to elevate privileges below.")
        update_cmd.append("sudo")
        cmd.append("sudo")
    cmd += ["apt-get", "install"]
    if install_or_upgrade == "upgrade":
        cmd += "--upgrade"
    cmd += ["-y", dep]
    update_cmd += ["apt-get", "upgrade"]
    subprocess.run(update_cmd, check=True, shell=False)  # noqa: S603
    subprocess.run(cmd, check=True, shell=False)  # noqa: S603


def install_apt_group_dependency(
    dependency_group: AptDependencyGroupsType,
    install_or_upgrade: Literal["install", "upgrade"],
) -> None:
    """Installs all dependencies for an apt dependency group."""
    all_deps = APT_GROUP_DEPENDENCIES[dependency_group]
    install_apt_dependency(all_deps, install_or_upgrade)


def install_pip_dependency(
    dep: str | list[str] | PipDependencyOptionsType | list[PipDependencyOptionsType],
    install_or_upgrade: Literal["install", "upgrade"],
    no_deps: bool,
) -> None:
    """Installs a single or multiple apt dependencies."""
    used_dep = " ".join(dep) if isinstance(dep, list) else str(dep)
    cmd = get_pip_env_cmd()
    if install_or_upgrade == "upgrade":
        cmd += ["--upgrade"]
    if no_deps:
        cmd += ["--no-deps"]
    cmd += [used_dep]
    subprocess.run(cmd, check=True, shell=False)  # noqa: S603


def install_pip_group_dependency(
    dependency_group: PipDependencyGroupOptionsType,
    install_or_upgrade: Literal["install", "upgrade"],
    no_deps: bool,
) -> None:
    """Installs all dependencies for a pip dependency group."""
    dependencies = PIP_GROUP_DEPENDENCIES[dependency_group]
    install_pip_dependency(dependencies, install_or_upgrade, no_deps)
