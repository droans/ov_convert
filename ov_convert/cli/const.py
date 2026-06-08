"""CLI Constants."""

import typing
from typing import Literal

PipDependencyOptionsType = Literal[
    "openvino",
    "openvino-genai",
    "openvino-tokenizers",
    "transformers",
    "optimum",
    "optimum-intel",
    "optimum-onnx",
]

PIP_DEPENDENCY_OPTIONS: list[PipDependencyOptionsType] = list(
    typing.get_args(PipDependencyOptionsType),
)

PipDependencyGroupOptionsType = Literal[
    "all-pip",
    "all-ov",
    "all-optimum",
]
PIP_DEPENDENCY_GROUP_OPTIONS: list[PipDependencyGroupOptionsType] = list(
    typing.get_args(PipDependencyGroupOptionsType),
)

AptDependencyGroupsType = Literal["all-compute"]
APT_DEPENDENCY_GROUP_OPTIONS: list[AptDependencyGroupsType] = list(
    typing.get_args(AptDependencyGroupsType),
)

APT_GROUP_DEPENDENCIES: dict[AptDependencyGroupsType, list[str]] = {
    "all-compute": [
        "intel-igc-core",
        "intel-igc-opencl",
        "intel-ocloc",
        "intel-opencl-icd",
        "libze-intel-gpu1",
        "libigdgmm12",
        "libze-dev",
        "libze1",
    ],
}

OV_DEPENDENCIES: list[PipDependencyOptionsType] = [
    "openvino",
    "openvino-genai",
    "openvino-tokenizers",
]
OPTIMUM_DEPENDENCIES: list[PipDependencyOptionsType] = [
    "optimum",
    "optimum-intel",
    "optimum-onnx",
]


PIP_GROUP_DEPENDENCIES: dict[PipDependencyGroupOptionsType, list[PipDependencyOptionsType]] = {
    "all-ov": OV_DEPENDENCIES,
    "all-optimum": OPTIMUM_DEPENDENCIES,
    "all-pip": PIP_DEPENDENCY_OPTIONS,
}

PIP_REPOS: dict[PipDependencyOptionsType, str] = {
    "openvino": "openvinotoolkit/openvino",
    "openvino-genai": "openvinotoolkit/openvino-genai",
    "openvino-tokenizers": "openvinotoolkit/openvino-tokenizers",
    "transformers": "huggingface/transformers",
    "optimum": "huggingface/optimum",
    "optimum-intel": "huggingface/optimum-intel",
    "optimum-onnx": "huggingface/optimum-onnx",
}
