# ov-convert

Utility to convert LLMs/VLMs to OpenVINO format using YAML config files or configuration passed via Python scripts.

## Features:
* Supports all options supported by optimum-intel/nncf
* Reproducible - By using configuration files, you can keep accurate records of the configuration settings applied to each model. This utility also supports outputting the passed config to file, allowing you to keep record of any exports performed programmatically.
* Has both a CLI interface (`ov-convert`) and programmatic interface.


## Usage:

See [example_config.yaml](example_config.yaml) for configuration example. This file is not exhaustive - more options can be found in the [src/models](src/models) folder.

### CLI usage:

```bash
ov-convert conversion_config.yaml
```

### Programmatic usage:

#### Convert from config file
```python
from ov_convert import convert

convert.export_from_config_file("path/to/config/file.yaml")
```

#### Convert from configuration:
```python
from ov_convert import export, models

model = models.LlmModelConfiguration(...) # Create model programmatically
export(model)
```