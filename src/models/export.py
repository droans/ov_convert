"""Model export settings config schema."""
from pydantic import BaseModel, DirectoryPath


class BaseModelExportSettingsConfigSchema(BaseModel):
  """Base config schema for model export settings."""

  configuration: bool = False
  include_defaults: bool = False
  tokenizer: bool = True
  processor: bool = True
  preprocessor: bool = True
  model: bool = True
  path: DirectoryPath

class LlmModelExportSettingsConfigSchema(BaseModelExportSettingsConfigSchema):
  """Config schema for LLM model export settings."""

class VlmModelExportSettingsConfigSchema(BaseModelExportSettingsConfigSchema):
  """Config schema for VLM model export settings."""

  preprocessor: bool = True
