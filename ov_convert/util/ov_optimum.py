"""OpenVINO/Optimum Utilities."""

from nncf import IgnoredScope
from optimum.intel import (
    OVModelForCausalLM,
    OVModelForVisualCausalLM,
    OVPipelineQuantizationConfig,
    OVQuantizationConfig,
    OVWeightQuantizationConfig,
)
from optimum.intel.openvino.modeling_base import OVBaseModel

from ov_convert.models import FullQuantizationConfig, WeightQuantizationConfig
from ov_convert.models.model import ModelConfigurationInternal
from ov_convert.models.quantization import (
    LlmQuantizationSettingsSchema,
    VlmQuantizationSettingsSchema,
)


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
    load_opts = conf.load_options.model_dump()
    model_name = conf.model.name
    model_type = conf.model.type
    quant_config = generate_ov_pipeline_config(conf.quantization)
    processor = conf.quantization.processor
    tokenizer = conf.quantization.tokenizer

    cls = OVModelForVisualCausalLM if model_type == "vlm" else OVModelForCausalLM
    return cls.from_pretrained(
        model_name,
        export=True,
        compile=False,
        quantization_config=quant_config,
        processor=processor,
        tokenizer=tokenizer,
        **load_opts,
    )
