"""Converts model."""

import json
import logging
from pathlib import Path

import yaml
from nncf import IgnoredScope
from openvino import save_model
from openvino_tokenizers import convert_tokenizer
from optimum.intel import (
    OVModelForCausalLM,
    OVModelForVisualCausalLM,
    OVPipelineQuantizationConfig,
    OVQuantizationConfig,
    OVWeightQuantizationConfig,
)
from optimum.intel.openvino.modeling_base import OVBaseModel
from transformers import (
    AutoImageProcessor,
    AutoProcessor,
    AutoTokenizer,
    TokenizersBackend,
)

from ov_convert.models import LlmModelConfiguration, VlmModelConfiguration
from ov_convert.models.model import ModelConfigurationInternal
from ov_convert.models.quantization import (
    FullQuantizationConfig,
    LlmQuantizationSettingsSchema,
    VlmQuantizationSettingsSchema,
    WeightQuantizationConfig,
)

logger = logging.getLogger()


def get_config_from_file(config_file_path: str) -> ModelConfigurationInternal:
    """Loads the data from `config_file_path` into the ModelConfigurationInternal model."""
    with open(config_file_path) as f:
        model = ModelConfigurationInternal(config=yaml.safe_load(f.read()))
    if not model.config.export.path:
        model.config.export.path = Path(config_file_path).parent
    return model


def dump_config_to_file(config: VlmModelConfiguration | LlmModelConfiguration) -> None:
    """Dumps the configuration to a file."""
    include_defaults = config.export.include_defaults
    msg = f"⚠️[OVConvert]⚠️ Include defaults: {include_defaults}"
    logger.error(msg)

    # Must dump to json and reload due to PosixPath objects in schema
    data = json.loads(config.model_dump_json(exclude_unset=not include_defaults))

    yaml_str = yaml.safe_dump(data)
    save_path = f"{config.export.path}/conversion_config.yaml"
    with open(save_path, "w") as f:
        f.write(yaml_str)


def export_from_config_file(config_file_path: str) -> None:
    """Exports a model using the config file passed."""
    export(get_config_from_file(config_file_path).config)


def export(config: VlmModelConfiguration | LlmModelConfiguration) -> None:
    """Exports a model and its components using the passed configuration."""
    model_config = ModelConfigurationInternal(config=config)
    export_conf = model_config.config.export
    export_tokenizer = export_conf.tokenizer
    export_processor = export_conf.processor
    export_preprocessor = (
        isinstance(config, VlmModelConfiguration) and export_conf.preprocessor
    )
    _export_model = export_conf.model
    export_config = export_conf.configuration
    msg = f"⚠️[OVConvert]⚠️ Exporting {config.model.name}"
    logger.info(msg)

    # Export tokenizers, processors first. If they have an issue, we should
    if export_tokenizer:
        logger.info("⚠️[OVConvert]⚠️ Exporting tokenizer.")
        export_model_tokenizer(model_config)
        logger.info("⚠️[OVConvert]⚠️ Tokenizer exported.")
    if export_processor:
        logger.info("⚠️[OVConvert]⚠️ Exporting processor.")
        export_model_processor(model_config)
        logger.info("⚠️[OVConvert]⚠️ Processor exported.")
    if export_preprocessor:
        logger.info("⚠️[OVConvert]⚠️ Exporting preprocessor.")
        export_model_preprocessor(model_config)
        logger.info("⚠️[OVConvert]⚠️ Preprocessor exported.")
    if _export_model:
        logger.info("⚠️[OVConvert]⚠️ Exporting model.")
        export_model(model_config)
        logger.info("⚠️[OVConvert]⚠️ Model exported.")
    if export_config:
        logger.info("⚠️[OVConvert]⚠️ Exporting config.")
        dump_config_to_file(config)
        logger.info("⚠️[OVConvert]⚠️ Config exported.")


def export_model(model_config: ModelConfigurationInternal) -> None:
    """Converts a model."""
    save_dir = model_config.config.export.path
    model = load_model(model_config)
    assert save_dir
    model.save_pretrained(save_dir)


def export_model_tokenizer(model_config: ModelConfigurationInternal) -> None:
    """Exports the model tokenizer."""
    conf = model_config.config
    model_name = conf.model.name
    trust_remote_code = conf.load_options.trust_remote_code
    save_dir = conf.export.path
    hf_tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=trust_remote_code
    )
    ov_tokenizer, ov_detokenizer = convert_tokenizer(
        hf_tokenizer, with_detokenizer=True
    )
    tokenizer_save_path = f"{save_dir}/openvino_tokenizer.xml"
    detokenizer_save_path = f"{save_dir}/openvino_detokenizer.xml"
    save_model(ov_tokenizer, tokenizer_save_path)
    save_model(ov_detokenizer, detokenizer_save_path)


def export_model_processor(model_config: ModelConfigurationInternal) -> None:
    """Exports the model processor."""
    conf = model_config.config
    model_name = conf.model.name
    trust_remote_code = conf.load_options.trust_remote_code
    save_dir = conf.export.path
    processor: TokenizersBackend = AutoProcessor.from_pretrained(
        model_name, trust_remote_code=trust_remote_code
    )
    assert save_dir
    processor.save_pretrained(save_dir)


def export_model_preprocessor(model_config: ModelConfigurationInternal) -> None:
    """Exports the model pre-processor."""
    conf = model_config.config
    model_name = conf.model.name
    trust_remote_code = conf.load_options.trust_remote_code
    save_dir = conf.export.path
    preprocessor: TokenizersBackend = AutoImageProcessor.from_pretrained(
        model_name, trust_remote_code=trust_remote_code
    )
    assert save_dir
    preprocessor.save_pretrained(save_dir)


def _generate_single_quant_config(
    quant_config: FullQuantizationConfig | WeightQuantizationConfig,
) -> OVWeightQuantizationConfig | OVQuantizationConfig:
    """Generates a single quantization config."""
    return generate_ov_component_quant_config(quant_config)


def generate_ov_pipeline_quant_config(
    quant_config: (
        dict[str, FullQuantizationConfig | WeightQuantizationConfig]
        | FullQuantizationConfig
        | WeightQuantizationConfig
    ),
) -> dict:
    """Generates the pipeline config for a model."""
    if isinstance(quant_config, dict):
        return {k: _generate_single_quant_config(v) for k, v in quant_config.items()}
    return {"lm_model": _generate_single_quant_config(quant_config)}


def generate_ov_component_quant_config(
    config: WeightQuantizationConfig | FullQuantizationConfig,
) -> OVWeightQuantizationConfig | OVQuantizationConfig:
    """Generates the quantization config for a component in the model."""
    kwargs = config.kwargs
    passed_config: dict = config.model_dump()
    passed_config.pop("kwargs")
    passed_config = {**passed_config, **(kwargs or {})}
    passed_config["ignored_scope"] = (
        IgnoredScope(**config.ignored_scope.model_dump())
        if config.ignored_scope
        else {}
    )
    cls = (
        OVWeightQuantizationConfig
        if isinstance(config, WeightQuantizationConfig)
        else OVQuantizationConfig
    )
    return cls(**passed_config)


def generate_ov_pipeline_config(
    quant_config: LlmQuantizationSettingsSchema | VlmQuantizationSettingsSchema,
) -> OVPipelineQuantizationConfig:
    """Generates the pipeline quantization config."""
    passed_config: dict = quant_config.model_dump()
    passed_config.pop("config")
    passed_config.pop("default_config")
    passed_config["quantization_configs"] = (
        {}
        if not quant_config.config
        else generate_ov_pipeline_quant_config(quant_config.config)
    )
    passed_config["default_config"] = (
        {}
        if not quant_config.default_config
        else _generate_single_quant_config(quant_config.default_config)
    )
    return OVPipelineQuantizationConfig(**passed_config)


def load_model(model_config: ModelConfigurationInternal) -> OVBaseModel:
    """Loads a model using the passed configuration."""
    conf = model_config.config
    model_name = conf.model.name
    model_type = conf.model.type
    trust_remote_code = conf.load_options.trust_remote_code
    device = conf.load_options.device or "auto"
    ov_config = conf.load_options.ov_config
    quant_config = generate_ov_pipeline_config(conf.quantization)
    processor = conf.quantization.processor
    tokenizer = conf.quantization.tokenizer

    cls = OVModelForVisualCausalLM if model_type == "vlm" else OVModelForCausalLM
    return cls.from_pretrained(
        model_name,
        export=True,
        trust_remote_code=trust_remote_code,
        device=device,
        ov_config=ov_config,
        compile=False,
        quantization_config=quant_config,
        processor=processor,
        tokenizer=tokenizer,
    )
