"""Model information config schema."""
from typing import Literal

from pydantic import BaseModel


class BaseModelInformationConfig(BaseModel):
  """Base Model information config."""

  name: str

class VlmModelInformationConfig(BaseModelInformationConfig):
  """Information Configuration for VLMs."""

  type: Literal["vlm"]

class LlmModelInformationConfig(BaseModelInformationConfig):
  """Information Configuration for LLMs."""

  type: Literal["llm"]
