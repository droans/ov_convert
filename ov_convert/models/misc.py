"""Misc model settings config schema."""

from typing import Literal

from pydantic import BaseModel, DirectoryPath, NewPath

from ov_convert.models.const import (
    DEFAULT_LOG_FILTER_COMPONENTS,
    DEFAULT_LOG_FORMAT,
    DEFAULT_LOG_LEVEL,
    LogLevelsType,
)


#################################
###      Model Information     ##
#################################
class BaseModelInformationConfig(BaseModel):
    """Base Model information config."""

    name: str


class VlmModelInformationConfig(BaseModelInformationConfig):
    """Information Configuration for VLMs."""

    type: Literal["vlm"]


class LlmModelInformationConfig(BaseModelInformationConfig):
    """Information Configuration for LLMs."""

    type: Literal["llm"]


class LoadOptions(BaseModel):
    """Load options config schema."""

    force_download: bool = False
    local_files_only: bool = False
    revision: str | None = None
    device: str = "auto"
    cache_dir: DirectoryPath | str | None = None
    subfolder: str = ""
    trust_remote_code: bool = False
    ov_config: dict = {}


class BaseModelExportSettingsConfigSchema(BaseModel):
    """Base config schema for model export settings."""

    configuration: bool = False
    include_defaults: bool = False
    tokenizer: bool = True
    processor: bool = True
    preprocessor: bool = True
    model: bool = True
    path: NewPath | DirectoryPath | None = None


class LlmModelExportSettingsConfigSchema(BaseModelExportSettingsConfigSchema):
    """Config schema for LLM model export settings."""


class VlmModelExportSettingsConfigSchema(BaseModelExportSettingsConfigSchema):
    """Config schema for VLM model export settings."""

    preprocessor: bool = True


class LogFilters(BaseModel):
    """Config schema for log filters."""

    components: list[str] = DEFAULT_LOG_FILTER_COMPONENTS
    messages: list[str] = []


class LogConfigSchema(BaseModel):
    """Config schema for log settings."""

    level: LogLevelsType = DEFAULT_LOG_LEVEL
    format: str = DEFAULT_LOG_FORMAT
    path: str | None = None
    filters: LogFilters = LogFilters()
