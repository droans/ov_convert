"""OpenVINO/Optimum Utilities."""

import optimum.intel
from nncf import IgnoredScope

from ov_convert.models import FullQuantizationConfig, WeightQuantizationConfig
from ov_convert.models.model import ModelConfigurationInternal
from ov_convert.models.quantization import (
    LlmQuantizationSettingsSchema,
    VlmCustomDataset,
    VlmQuantizationSettingsSchema,
)
from ov_convert.util.log import logger


def _generate_single_quant_config(
    quant_config: FullQuantizationConfig | WeightQuantizationConfig,
) -> optimum.intel.OVWeightQuantizationConfig | optimum.intel.OVQuantizationConfig:
    """Generate a single quantization config."""
    return generate_ov_component_quant_config(quant_config)


def generate_ov_pipeline_quant_config(
    quant_config: (
        dict[str, FullQuantizationConfig | WeightQuantizationConfig]
        | FullQuantizationConfig
        | WeightQuantizationConfig
    ),
) -> dict:
    """Generate the pipeline config for a model."""
    if isinstance(quant_config, dict):
        return {k: _generate_single_quant_config(v) for k, v in quant_config.items()}
    return {"lm_model": _generate_single_quant_config(quant_config)}


def generate_ov_component_quant_config(
    config: WeightQuantizationConfig | FullQuantizationConfig,
) -> optimum.intel.OVWeightQuantizationConfig | optimum.intel.OVQuantizationConfig:
    """Generate the quantization config for a component in the model."""
    kwargs = config.kwargs
    passed_config: dict = config.model_dump()
    if passed_config["ignored_scope"] is not None:
        validate = passed_config["ignored_scope"].pop("validate_scopes")
        passed_config["ignored_scope"]["validate"] = validate
    if isinstance(config.dataset, VlmCustomDataset):
        monkeypatch_custom_dataset(config.dataset)
        passed_config["dataset"] = config.dataset.id
    passed_config.pop("kwargs")
    passed_config = {**passed_config, **(kwargs or {})}
    passed_config["ignored_scope"] = (
        IgnoredScope(**passed_config["ignored_scope"]) if config.ignored_scope else {}
    )
    cls = (
        optimum.intel.OVWeightQuantizationConfig
        if isinstance(config, WeightQuantizationConfig)
        else optimum.intel.OVQuantizationConfig
    )
    return cls(**passed_config)


def generate_ov_pipeline_config(
    quant_config: LlmQuantizationSettingsSchema | VlmQuantizationSettingsSchema,
) -> optimum.intel.OVPipelineQuantizationConfig:
    """Generate the pipeline quantization config."""
    passed_config: dict = quant_config.model_dump()
    passed_config.pop("config")
    passed_config.pop("default_config")
    passed_config["quantization_configs"] = (
        {} if not quant_config.config else generate_ov_pipeline_quant_config(quant_config.config)
    )
    passed_config["default_config"] = (
        {}
        if not quant_config.default_config
        else _generate_single_quant_config(quant_config.default_config)
    )
    if isinstance(quant_config.dataset, VlmCustomDataset):
        monkeypatch_custom_dataset(quant_config.dataset)
        passed_config["dataset"] = quant_config.dataset.id
    return optimum.intel.OVPipelineQuantizationConfig(**passed_config)


def load_model(
    model_config: ModelConfigurationInternal,
) -> optimum.intel.openvino.modeling_base.OVBaseModel:  # ty:ignore[possibly-missing-submodule]
    """Load a model using the passed configuration."""
    conf = model_config.config
    load_opts = conf.load_options.model_dump()
    model_name = conf.model.name
    model_type = conf.model.type
    quant_config = generate_ov_pipeline_config(conf.quantization)
    processor = conf.quantization.processor
    tokenizer = conf.quantization.tokenizer

    cls = (
        optimum.intel.OVModelForVisualCausalLM
        if model_type == "vlm"
        else optimum.intel.OVModelForCausalLM
    )
    return cls.from_pretrained(
        model_name,
        export=True,
        compile=False,
        quantization_config=quant_config,
        processor=processor,
        tokenizer=tokenizer,
        **load_opts,
    )


def monkeypatch_custom_dataset(dataset_config: VlmCustomDataset) -> None:
    """Monkey patches optimum to support a custom dataset."""
    logger.info(
        f"Got custom dataset {dataset_config.id}, monkey patching optimum-intel to add support...",
    )
    dataset = dataset_config.model_dump()
    dataset["streaming"] = True
    added_dataset = {dataset_config.id: dataset}
    original_dataset = optimum.intel.openvino.utils.PREDEFINED_VISUAL_LM_DATASETS
    if dataset_config.id in original_dataset:
        logger.info(f"Dataset {dataset_config.id} already patched in, returning.")
        return
    logger.info(f"Adding {added_dataset}")
    datasets = {
        **optimum.intel.openvino.utils.PREDEFINED_VISUAL_LM_DATASETS,
        **added_dataset,
    }
    logger.info(f"All datasets: {datasets}")
    optimum.intel.openvino.utils.PREDEFINED_VISUAL_LM_DATASETS = datasets
    optimum.intel.openvino.configuration.PREDEFINED_VISUAL_LM_DATASETS = datasets
    optimum.intel.openvino.quantization.PREDEFINED_VISUAL_LM_DATASETS = datasets
    logger.info("Dataset monkey patched in.")
