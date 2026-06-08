# ov-convert

Utility to convert LLMs/VLMs to OpenVINO format using YAML config files or configuration passed via Python scripts.

## Features:
* Supports all options supported by optimum-intel/nncf
* Reproducible - By using configuration files, you can keep accurate records of the configuration settings applied to each model. This utility also supports outputting the passed config to file, allowing you to keep record of any exports performed programmatically.
* Has both a CLI interface (`ov-convert`) and programmatic interface.
* Allows for updating apt and pip dependencies

## Installing

This project can be installed via `pip`:

```sh
pip install git+https://github.com/droans/ov_convert@v0.2.0
```

Update `v0.2.0` with the version you intend to use.

If you have issues with dependencies (IE - `optimum` is often unhappy with the version of `transformers` required), add `--no-deps` to the end of the command. You may need to install the dependencies manually - see [pyproject.toml](pyproject.toml) for the requirements.


## Usage:


```
                                ov-convert
Usage:
    Convert a model using a config file:
        ov-convert [convert] config.yaml

    Install/Update OpenVINO dependencies:
        ov-convert dep [upgrade|install] [dependency|dependency-group]
            Dependencies [group(s)]:
                * openvino [all-ov, all-pip, all]
                * openvino-genai [all-ov, all-pip, all]
                * openvino-tokenizers [all-ov, all-pip, all]
                * optimum [all-optimum, all-pip, all]
                * optimum-intel [all-optimum, all-pip, all]
                * optimum-onnx [all-optimum, all-pip, all]
                * transformers [all-pip, all]
                * IGC/ICR/L0 [all-compute, all]
                    NOTE:   Intel Graphics Compiler, Intel Compute Runtime,
                            and Level Zero packages are shared dependencies
                            and cannot be installed individually.
Parameters:
    ov-convert deps:
        When upgrading/installing individual pip dependencies,
        the following parameters are available:
            -b, --branch            Specify the branch used by the dependency
            -pr, --pull-request     Specify a specific pull request to use for
                                    the dependency
```

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
