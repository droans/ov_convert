"""Models for Quantization Settings."""
from pydantic import BaseModel, FilePath, ValidationError, field_validator

from src.const import (
  QUANT_BACKUP_PRECISION_TYPE,
  QUANT_DEFAULT_BACKUP_PRECISION,
  QUANT_DEFAULT_GROUP_SIZE_FALLBACK,
  QUANT_DEFAULT_SENSITIVITIY_METRIC,
  QUANT_GROUP_SIZE_FALLBACK_TYPE,
  QUANT_SENSITIVITIY_METRIC_TYPE,
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
  backup_precision: QUANT_BACKUP_PRECISION_TYPE | None = QUANT_DEFAULT_BACKUP_PRECISION
  dq_group_size: int | None = None
  gptq: bool = False
  group_size: int = 128
  group_size_fallback: QUANT_GROUP_SIZE_FALLBACK_TYPE | None = QUANT_DEFAULT_GROUP_SIZE_FALLBACK
  lora_correction: bool = False
  ratio: float | None = 1.0
  scale_estimation: bool
  sensitivity_metric: QUANT_SENSITIVITIY_METRIC_TYPE | None = QUANT_DEFAULT_SENSITIVITIY_METRIC
  statistics_path: FilePath

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
