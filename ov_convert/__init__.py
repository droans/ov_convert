"""OpenVINO Conversion Utility."""

import sys

load_as_script = sys.argv and sys.argv[0].endswith("ov-convert")
if not load_as_script:
    from ov_convert import models

    from .convert import export, export_from_config_file

    __all__ = (
        "export",
        "export_from_config_file",
        "models",
    )
