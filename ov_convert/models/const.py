"""Constants."""

from typing import Annotated, Literal

from annotated_types import Ge, Le
from pydantic import constr

BACKUP_PRECISION_TYPE = Literal["int8_sym", "int8_asym"]

GROUP_SIZE_FALLBACK_TYPE = Literal["error", "ignore", "adjust"]

SENSITIVITIY_METRIC_TYPE = Literal[
    "weight_quantization_error",
    "hessian_input_activation",
    "mean_activation_variance",
    "max_activation_variance",
    "mean_activation_magnitude",
]

QUANT_METHOD_TYPE = Literal["awq", "hybrid", "default"]
DEFAULT_QUANT_METHOD_TYPE: QUANT_METHOD_TYPE = "default"

LLM_DATASETS = [
    "auto",
    "wikitext2",
    "c4",
    "c4-new",
    "gsm8k",
]
LLM_DATASET_TYPE = constr(pattern=rf"({'|'.join(LLM_DATASETS)})(:seq_len=\d+)?$")
VLM_DATASETS = ["contextual"]
VLM_DATASET_TYPE = constr(pattern=rf"({'|'.join(VLM_DATASETS)})(:seq_len=\d+)?$")

WEIGHT_FORMATS_TYPE = Literal[
    "fp32",
    "fp16",
    "int8",
    "int4",
    "mxfp4",
    "nf4",
    "cb4",
]

QUANT_MODES_TYPE = Literal[
    "int8",
    "f8e4m3",
    "f8e5m2",
    "int8_f8e4m3",
    "int4_f8e4m3",
    "int4_f8e5m2",
]

PERCENTAGE_TYPE = Annotated[float, Ge(0), Le(1)]
