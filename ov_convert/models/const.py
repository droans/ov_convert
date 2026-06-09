"""Constants."""

import typing
from typing import Annotated, Literal

from annotated_types import Ge, Le
from pydantic import constr

BackupPrecisionType = Literal["int8_sym", "int8_asym"]

GroupSizeFallbackType = Literal["error", "ignore", "adjust"]

SensitivityMetricType = Literal[
    "weight_quantization_error",
    "hessian_input_activation",
    "mean_activation_variance",
    "max_activation_variance",
    "mean_activation_magnitude",
]

QuantMethodType = Literal["awq", "hybrid", "default"]
DEFAULT_QUANT_METHOD_TYPE: QuantMethodType = "default"

LLM_DATASETS = [
    "auto",
    "wikitext2",
    "c4",
    "c4-new",
    "gsm8k",
]
LlmDatasetType = constr(pattern=rf"({'|'.join(LLM_DATASETS)})(:seq_len=\d+)?$")
VLM_DATASETS = ["contextual"]
VlmDatasetType = constr(pattern=rf"({'|'.join(VLM_DATASETS)})(:seq_len=\d+)?$")

WeightFormatsType = Literal[
    "fp32",
    "fp16",
    "int8",
    "int4",
    "mxfp4",
    "nf4",
    "cb4",
]

FullQuantFormatsType = Literal[
    "int8",
    "f8e4m3",
    "f8e5m2",
]

QuantModesType = Literal[
    "int8",
    "f8e4m3",
    "f8e5m2",
    "int8_f8e4m3",
    "int4_f8e4m3",
    "int4_f8e5m2",
]

PercentageType = Annotated[float, Ge(0), Le(1)]

LogLevelsType = Literal[
    "debug",
    "DEBUG",
    "info",
    "INFO",
    "warn",
    "WARN",
    "error",
    "ERROR",
    "critical",
    "CRITICAL",
    "fatal",
    "FATAL",
]
LOG_LEVELS = list(typing.get_args(LogLevelsType))
DEFAULT_LOG_LEVEL: LogLevelsType = "error"
DEFAULT_LOG_FORMAT = "%(asctime)s | %(levelname)s | %(message)s"
DEFAULT_LOG_PATH = "./ov_convert.log"
DEFAULT_LOG_FILTER_COMPONENTS = [
    "httpx",
    "httpcore.connection",
    "httpcore.http11",
    "filelock",
]
