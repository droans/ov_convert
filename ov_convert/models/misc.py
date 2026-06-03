"""Misc model settings config schema."""

from typing import Literal

from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE
from pydantic import BaseModel, DirectoryPath


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
    device: str | None = "auto"
    cache_dir: DirectoryPath | str = HUGGINGFACE_HUB_CACHE
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
    path: DirectoryPath | None = None


class LlmModelExportSettingsConfigSchema(BaseModelExportSettingsConfigSchema):
    """Config schema for LLM model export settings."""


class VlmModelExportSettingsConfigSchema(BaseModelExportSettingsConfigSchema):
    """Config schema for VLM model export settings."""

    preprocessor: bool = True


class BaseModelExportSettingsConfigSchema(BaseModel):
    """Base config schema for model export settings."""

    configuration: bool = False
    include_defaults: bool = False
    tokenizer: bool = True
    processor: bool = True
    preprocessor: bool = True
    model: bool = True
    path: DirectoryPath | None = None
