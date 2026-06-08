"""CLI Utility functions."""

import shutil


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
