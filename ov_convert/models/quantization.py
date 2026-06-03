# ty:ignore[invalid-type-form]  noqa: ERA001
"""Models for Quantization Settings."""
from typing import Literal

from pydantic import BaseModel, DirectoryPath

from .const import (
    BACKUP_PRECISION_TYPE,
    DEFAULT_QUANT_METHOD_TYPE,
    GROUP_SIZE_FALLBACK_TYPE,
    LLM_DATASET_TYPE,
    PERCENTAGE_TYPE,
    QUANT_METHOD_TYPE,
    SENSITIVITIY_METRIC_TYPE,
    VLM_DATASET_TYPE,
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
    dataset: VLM_DATASET_TYPE | LLM_DATASET_TYPE | None = None
    tokenizer: str | None = None
    processor: str | None = None
    dtype: WEIGHT_FORMATS_TYPE | None = None
    kwargs: dict | None = None


class FullQuantizationConfig(BaseQuantizationConfig):
    """Full quantization config for a model slice."""

    fast_bias_correction: bool = True
    overflow_fix: Literal["enable", "disable", "first_layer_only"] = "disable"
    smooth_quant_alpha: PERCENTAGE_TYPE | Literal[-1] | None = None


class WeightQuantizationConfig(BaseQuantizationConfig):
    """Weights-only quantization config for a model slice."""

    group_size: int | None = None
    ratio: PERCENTAGE_TYPE = 1.0
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


class BaseQuantizationSettingsSchema(BaseModel):
    """Base quantization config schema."""

    num_samples: int | None = None
    tokenizer: str | None = None
    processor: str | None = None
    default_config: FullQuantizationConfig | WeightQuantizationConfig | None = (
        WeightQuantizationConfig()
    )


class LlmQuantizationSettingsSchema(BaseQuantizationSettingsSchema):
    """LLM Quantization Settings."""

    config: FullQuantizationConfig | WeightQuantizationConfig | None = None
    dataset: LLM_DATASET_TYPE | None = None


class VlmQuantizationSettingsSchema(BaseQuantizationSettingsSchema):
    """VLM Quantization Settings."""

    config: dict[str, FullQuantizationConfig | WeightQuantizationConfig] | None = None
    dataset: VLM_DATASET_TYPE | None = None
