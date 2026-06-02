"""Models for language models."""
from pydantic import BaseModel

from src.models.export import LlmModelExportSettingsConfigSchema, VlmModelExportSettingsConfigSchema
from src.models.load_options import LoadOptions
from src.models.model_information import LlmModelInformationConfig, VlmModelInformationConfig
from src.models.quantization import LlmQuantizationSettingsSchema, VlmQuantizationSettingsSchema


class LlmModelConfiguration(BaseModel):
  """Config schema for LLMs."""

  model: LlmModelInformationConfig
  export: LlmModelExportSettingsConfigSchema
  quantization: LlmQuantizationSettingsSchema = LlmQuantizationSettingsSchema()
  load_options: LoadOptions = LoadOptions()

class VlmModelConfiguration(BaseModel):
  """Config schema for VLMs."""

  model: VlmModelInformationConfig
  export: VlmModelExportSettingsConfigSchema
  quantization: VlmQuantizationSettingsSchema = VlmQuantizationSettingsSchema()
  load_options: LoadOptions = LoadOptions()

class ModelConfigurationInternal(BaseModel):
  """Internal configuration format for model."""

  config: LlmModelConfiguration | VlmModelConfiguration
