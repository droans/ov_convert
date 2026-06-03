"""Models for language models."""

from pydantic import BaseModel

from ov_convert.models.misc import (
    LlmModelExportSettingsConfigSchema,
    LlmModelInformationConfig,
    LoadOptions,
    LogConfigSchema,
    VlmModelExportSettingsConfigSchema,
    VlmModelInformationConfig,
)
from ov_convert.models.quantization import (
    LlmQuantizationSettingsSchema,
    VlmQuantizationSettingsSchema,
)


class LlmModelConfiguration(BaseModel):
    """Config schema for LLMs."""

    model: LlmModelInformationConfig
    export: LlmModelExportSettingsConfigSchema = LlmModelExportSettingsConfigSchema()
    log: LogConfigSchema = LogConfigSchema()
    quantization: LlmQuantizationSettingsSchema = LlmQuantizationSettingsSchema()
    load_options: LoadOptions = LoadOptions()


class VlmModelConfiguration(BaseModel):
    """Config schema for VLMs."""

    model: VlmModelInformationConfig
    export: VlmModelExportSettingsConfigSchema = VlmModelExportSettingsConfigSchema()
    log: LogConfigSchema = LogConfigSchema()
    quantization: VlmQuantizationSettingsSchema = VlmQuantizationSettingsSchema()
    load_options: LoadOptions = LoadOptions()


class ModelConfigurationInternal(BaseModel):
    """Internal configuration format for model."""

    config: LlmModelConfiguration | VlmModelConfiguration
