# ty:ignore[invalid-type-form]  noqa: ERA001
"""Models for Quantization Settings."""

from typing import Literal

from pydantic import BaseModel, DirectoryPath

from .const import (
    DEFAULT_QUANT_METHOD_TYPE,
    BackupPrecisionType,
    FullQuantFormatsType,
    GroupSizeFallbackType,
    LlmDatasetType,
    PercentageType,
    QuantMethodType,
    SensitivityMetricType,
    VlmDatasetType,
    WeightFormatsType,
)


class IgnoredScopeConfig(BaseModel):
    """Config for ignored scopes."""

    names: list[str] = []
    patterns: list[str] = []
    types: list[str] = []
    subgraphs: list[str] = []
    validate_scopes: bool = True


class VlmDatasetInputs(BaseModel):
    """Config schema for VLM custom dataset inputs."""

    image_url: str
    instruction: str


class VlmCustomDataset(BaseModel):
    """Base config for custom datasets."""

    id: str
    split: str
    inputs: VlmDatasetInputs


class BaseQuantizationConfig(BaseModel):
    """Base quantization config for a model slice."""

    bits: int = 8
    sym: bool = False
    ignored_scope: IgnoredScopeConfig | None = None
    num_samples: int | None = None
    dataset: VlmDatasetType | LlmDatasetType | None | VlmCustomDataset = None
    tokenizer: str | None = None
    processor: str | None = None
    kwargs: dict | None = None


class FullQuantizationConfig(BaseQuantizationConfig):
    """Full quantization config for a model slice."""

    fast_bias_correction: bool = True
    overflow_fix: Literal["enable", "disable", "first_layer_only"] = "disable"
    smooth_quant_alpha: PercentageType | Literal[-1] | None = None
    weights_only: Literal[False] = False  # Used to ensure selection of FullQuantizationConfig
    dtype: FullQuantFormatsType | None = None


class WeightQuantizationConfig(BaseQuantizationConfig):
    """Weights-only quantization config for a model slice."""

    group_size: int | None = None
    ratio: PercentageType = 1.0
    all_layers: bool | None = None
    sensitivity_metric: SensitivityMetricType | None = None
    quant_method: QuantMethodType = DEFAULT_QUANT_METHOD_TYPE
    scale_estimation: bool | None = None
    gptq: bool | None = None
    lora_correction: bool | None = None
    backup_precision: BackupPrecisionType | None = None
    statistics_path: DirectoryPath | None = None
    group_size_fallback: GroupSizeFallbackType | None = None
    dq_group_size: int | None = None
    weights_only: Literal[True] = True  # Used to ensure selection of WeightQuantizationConfig
    dtype: WeightFormatsType | None = None


class MixedQuantizationConfig(BaseModel):
    """Mixed quantization config for a model slice."""

    weight_quantization_config: WeightQuantizationConfig = WeightQuantizationConfig()
    full_quantization_config: FullQuantizationConfig = FullQuantizationConfig()
    ignored_scope: IgnoredScopeConfig | None = None
    num_samples: int | None = None
    dataset: VlmDatasetType | LlmDatasetType | None | VlmCustomDataset = None
    tokenizer: str | None = None
    processor: str | None = None
    kwargs: dict | None = None


QuantizationConfigType = WeightQuantizationConfig | FullQuantizationConfig | MixedQuantizationConfig


class BaseQuantizationSettingsSchema(BaseModel):
    """Base quantization config schema."""

    num_samples: int | None = None
    tokenizer: str | None = None
    processor: str | None = None
    default_config: QuantizationConfigType | None = WeightQuantizationConfig()


class LlmQuantizationSettingsSchema(BaseQuantizationSettingsSchema):
    """LLM Quantization Settings."""

    config: QuantizationConfigType | None = None
    dataset: LlmDatasetType | None = None


class VlmQuantizationSettingsSchema(BaseQuantizationSettingsSchema):
    """VLM Quantization Settings."""

    config: dict[str, QuantizationConfigType] | None = None
    dataset: VlmDatasetType | None | VlmCustomDataset = None
