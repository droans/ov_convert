"""Constants."""
import typing
from typing import Literal

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
VLM_DATASETS = ["contextual"]
