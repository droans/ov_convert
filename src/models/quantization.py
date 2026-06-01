"""Models for Quantization Settings."""
from pydantic import BaseModel, FilePath, ValidationError, field_validator

from src.const import (
  QUANT_DEFAULT_BACKUP_PRECISION,
  QUANT_DEFAULT_GROUP_SIZE_FALLBACK,
  QUANT_DEFAULT_SENSITIVITIY_METRIC,
  QUANT_OPTIONS_BACKUP_PRECISION,
  QUANT_OPTIONS_GROUP_SIZE_FALLBACK,
  QUANT_OPTIONS_SENSITIVITIY_METRIC,
)


class BaseQuantizationConfig(BaseModel):
  """Base quantization settings."""

  bits: int = 8
  dataset: str | None = None
  dtype: str | None
  ignored_scope: list[str] | None = None
  num_samples: int | None = None
  processor: str | None = None
  quant_method: str | None = None
  sym: bool = False
  tokenizer: str | None = None

class FullQuantizationConfig(BaseQuantizationConfig):
  """Quantization settings when applying full model quantization."""

  fast_bias_correction: bool
  model_type: str
  overflow_fix: bool
  smooth_quant_alpha: float

class WeightQuantizationConfig(BaseQuantizationConfig):
  """Quantization settings when applying weights-only quantization."""

  all_layers: bool | None
  backup_precision: str | None = QUANT_DEFAULT_BACKUP_PRECISION
  dq_group_size: int | None = None
  gptq: bool = False
  group_size: int = 128
  group_size_fallback: str | None = QUANT_DEFAULT_GROUP_SIZE_FALLBACK
  lora_correction: bool = False
  ratio: float | None = 1.0
  scale_estimation: bool
  sensitivity_metric: str | None = QUANT_DEFAULT_SENSITIVITIY_METRIC
  statistics_path: FilePath

  @field_validator("backup_precision")
  @classmethod
  def validate_backup_precision(cls, value: str | None) -> str | None:
    """Validates backup_precision."""
    if value not in QUANT_OPTIONS_BACKUP_PRECISION:
      msg = f"{value} is an invalid option for backup_precision. Expected one of {QUANT_OPTIONS_BACKUP_PRECISION}"
      raise ValidationError(msg)
    if value is None:
      return QUANT_DEFAULT_BACKUP_PRECISION
    return value

  @field_validator("group_size_fallback")
  @classmethod
  def validate_group_size_fallback(cls, value: str | None) -> str | None:
    """Validates group_size_fallback."""
    if value not in QUANT_OPTIONS_GROUP_SIZE_FALLBACK:
      msg = f"{value} is an invalid option for group_size_fallback. Expected one of {QUANT_OPTIONS_GROUP_SIZE_FALLBACK}"
      raise ValidationError(msg)
    if value is None:
      return QUANT_DEFAULT_GROUP_SIZE_FALLBACK
    return value

  @field_validator("sensitivity_metric")
  @classmethod
  def validate_sensitivity_metric(cls, value: str | None) -> str | None:
    """Validates sensitivity_metric."""
    if value not in QUANT_OPTIONS_SENSITIVITIY_METRIC:
      msg = f"{value} is an invalid option for sensitivity_metric. Expected one of {QUANT_OPTIONS_SENSITIVITIY_METRIC}"
      raise ValidationError(msg)
    if value is None:
      return QUANT_DEFAULT_SENSITIVITIY_METRIC
    return value

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
