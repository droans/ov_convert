"""Dependency Management CLI functions."""

from ov_convert.cli.const import (
    APT_DEPENDENCY_GROUP_OPTIONS,
    PIP_DEPENDENCY_GROUP_OPTIONS,
    PIP_DEPENDENCY_OPTIONS,
    PIP_REPOS,
)
from ov_convert.cli.model import DependencyManagementModelSchema
from ov_convert.cli.util import (
    install_apt_dependency,
    install_apt_group_dependency,
    install_pip_dependency,
    install_pip_group_dependency,
)


def cli_manage_dependencies(config: DependencyManagementModelSchema) -> None:
    """Manage dependencies of OpenVino."""
    print(config)
    if not test_config(config):
        return
    dependency = config.dependency
    if dependency in PIP_DEPENDENCY_OPTIONS:
        manage_pip_dependencies(config)
    elif dependency in PIP_DEPENDENCY_GROUP_OPTIONS:
        manage_pip_group_dependencies(config)
    elif dependency in APT_DEPENDENCY_GROUP_OPTIONS:
        manage_apt_group_dependencies(config)
    elif dependency == "all":
        manage_all_dependencies(config)
    return


def manage_all_dependencies(config: DependencyManagementModelSchema) -> None:
    """Manage all dependencies for OpenVino."""
    print("Installing ALL dependencies...")
    action = config.action
    for item in APT_DEPENDENCY_GROUP_OPTIONS:
        install_apt_group_dependency(item, action)
    install_pip_group_dependency("all-pip", action)


def manage_pip_dependencies(config: DependencyManagementModelSchema) -> None:
    """Manage Pip dependencies for OpenVino."""
    print("Installing pip dependency...")
    dep = config.dependency
    assert dep in PIP_DEPENDENCY_OPTIONS
    branch = config.branch
    pr = config.pr
    if branch and pr:
        msg = (
            "Error: Arguments -b/--branch and --pr/--pull-request are "
            "exclusive and cannot be used together."
        )
        raise SystemError(msg)
    if branch or pr:
        repo = PIP_REPOS[dep]
        arg = f"@git+https://github.com/{repo}"
    else:
        arg = ""
    if branch:
        arg += f"@{branch}"
        print(f"Branch arg: {arg}")
    if pr:
        arg += f"@refs/pull/{pr}/merge"
        print(f"PR arg: {arg}")
    cmd = dep + arg
    print(f"Using command: {cmd}")
    install_pip_dependency(f"{dep}{arg}", config.action)


def manage_pip_group_dependencies(config: DependencyManagementModelSchema) -> None:
    """Manage Pip group dependencies for OpenVino."""
    print("Installing pip dependency group...")
    install_pip_group_dependency(config.dependency, config.action)  # ty:ignore[invalid-argument-type]


def manage_apt_dependencies(config: DependencyManagementModelSchema) -> None:
    """Manage Apt dependencies for OpenVino."""
    print("Installing apt dependency...")
    install_apt_dependency(config.dependency, config.action)


def manage_apt_group_dependencies(config: DependencyManagementModelSchema) -> None:
    """Manage Apt group dependencies for OpenVino."""
    print("Installing apt dependency group...")
    install_apt_group_dependency(config.dependency, config.action)  # ty:ignore[invalid-argument-type]


def test_config(config: DependencyManagementModelSchema) -> bool:
    """Run config through all tests."""
    tests = [
        test_branch_pr,
    ]
    return all(test(config) for test in tests)


def test_branch_pr(config: DependencyManagementModelSchema) -> bool:
    """Test if branch/PR is improperly applied to apt/grouped dependencies."""
    if config.dependency in PIP_DEPENDENCY_OPTIONS:
        return True
    if config.branch:
        print("Error: -b, --b can only be used for individual pip dependencies.")
        return False
    if config.pr:
        print("Error: --pr, --pull-request can only be used for individual pip dependencies.")
        return False
    return True
