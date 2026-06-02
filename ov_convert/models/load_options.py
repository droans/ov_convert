"""Model load option config schema."""
from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE
from pydantic import BaseModel, DirectoryPath


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
