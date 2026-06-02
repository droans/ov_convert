"""Models for Quantization Settings."""
import re
from typing import Literal

from pydantic import BaseModel, DirectoryPath, ValidationError, field_validator

from .const import (
  BACKUP_PRECISION_TYPE,
  DEFAULT_QUANT_METHOD_TYPE,
  GROUP_SIZE_FALLBACK_TYPE,
  LLM_DATASETS,
  QUANT_METHOD_TYPE,
  SENSITIVITIY_METRIC_TYPE,
  VLM_DATASETS,
  WEIGHT_FORMATS_TYPE,
)


class IgnoredScopeConfig(BaseModel):
  """Config for ignored scopes."""

  names: list[str] = []
  patterns: list[str] = []
  types: list[str] = []
  subgraphs: list[str] = []
  validate: bool = True

class BaseQuantizationConfig(BaseModel):
  """Base quantization config for a model slice."""

  bits: int = 8
  sym: bool = False
  ignored_scope: IgnoredScopeConfig | None = None
  num_samples: int | None = None
  dataset: str | None = None
  tokenizer: str | None = None
  processor: str | None = None
  dtype: WEIGHT_FORMATS_TYPE | None = None
  kwargs: dict | None = None

class FullQuantizationConfig(BaseQuantizationConfig):
  """Full quantization config for a model slice."""

  fast_bias_correction: bool = True
  overflow_fix: Literal["enable", "disable", "first_layer_only"] = "disable"
  smooth_quant_alpha: float | None = None

  @field_validator("smooth_quant_alpha")
  @classmethod
  def validate_smooth_quant_alpha(cls, value: float | None) -> float | None:
    """Validates smooth_quant_alpha."""
    if value and (value != -1 or not 0 >= value >= 1):
      msg = f"smooth_quant_alpha must be between 0-1 or equal to -1, got {value} instead."
      raise ValidationError(msg)
    return value

class WeightQuantizationConfig(BaseQuantizationConfig):
  """Weights-only quantization config for a model slice."""

  group_size: int | None = None
  ratio: float = 1.0
  all_layers: bool | None = None
  sensitivity_metric: SENSITIVITIY_METRIC_TYPE | None = None
  quant_method: QUANT_METHOD_TYPE = DEFAULT_QUANT_METHOD_TYPE
  scale_estimation: bool | None = None
  gptq: bool | None = None
  lora_correction: bool | None = None
  backup_precision: BACKUP_PRECISION_TYPE | None = None
  statistics_path: DirectoryPath | None = None
  group_size_fallback: GROUP_SIZE_FALLBACK_TYPE | None = None
  dq_group_size: int | None = None

  @field_validator("ratio")
  @classmethod
  def validate_ratio(cls, value: int | None) -> int:
    """Validates ratio."""
    if value and value > 1.0:
      msg = f"ratio must be less than or equal to 1.0, received {value}."
      raise ValidationError(msg)
    if value is None:
      return 1
    return value

class BaseQuantizationSettingsSchema(BaseModel):
  """Base quantization config schema."""

  num_samples: int | None = None
  tokenizer: str | None = None
  processor: str | None = None
  dataset: str | None = None
  default_config: FullQuantizationConfig | WeightQuantizationConfig | None = None

  @classmethod
  def _validate_dataset(cls, datasets: list[str], value: str | None) -> bool:
    """Returns TRUE if dataset is valid."""
    joined = "|".join(datasets)
    regex = rf"({joined})(:seq_len=\d+)?$"
    return value is None or re.fullmatch(regex, value) is not None


class LlmQuantizationSettingsSchema(BaseQuantizationSettingsSchema):
  """LLM Quantization Settings."""

  config: FullQuantizationConfig | WeightQuantizationConfig | None = None

  @field_validator("dataset")
  @classmethod
  def validate_dataset(cls, value: str | None) -> str | None:
    """Validates argument for dataset is valid."""
    if not cls._validate_dataset(LLM_DATASETS, value):
      msg = f"`{value}` is invalid value for dataset. Expected `({'|'.join(LLM_DATASETS)})[:seq_len=XX]`"
      raise ValidationError(msg)


class VlmQuantizationSettingsSchema(BaseQuantizationSettingsSchema):
  """VLM Quantization Settings."""

  config: dict[str, FullQuantizationConfig | WeightQuantizationConfig] | None = None

  @field_validator("dataset")
  @classmethod
  def validate_dataset(cls, value: str | None) -> str | None:
    """Validates argument for dataset is valid."""
    if not cls._validate_dataset(VLM_DATASETS, value):
      msg = f"`{value}` is invalid value for dataset. Expected `({'|'.join(VLM_DATASETS)})[:seq_len=XX]`"
      raise ValidationError(msg)

