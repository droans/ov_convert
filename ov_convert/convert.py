"""Converts model."""

from typing import TYPE_CHECKING

from openvino import save_model
from openvino_tokenizers import convert_tokenizer
from optimum.intel.openvino.utils import (
    OV_DETOKENIZER_NAME,
    OV_TOKENIZER_NAME,
)
from transformers import (
    AutoImageProcessor,
    AutoProcessor,
    AutoTokenizer,
)

from ov_convert.models import LlmModelConfiguration, VlmModelConfiguration
from ov_convert.models.model import ModelConfigurationInternal
from ov_convert.util.fs import (
    create_directory_if_nonexistant,
    dump_config_to_file,
    get_config_from_file,
)
from ov_convert.util.log import logger, setup_logging
from ov_convert.util.ov_optimum import load_model

if TYPE_CHECKING:
    from transformers import TokenizersBackend

# Sub placeholders in consts, substituting them out.
_OV_TOKENIZER_NAME = OV_TOKENIZER_NAME.replace("{}", "")
_OV_DETOKENIZER_NAME = OV_DETOKENIZER_NAME.replace("{}", "")


def export_from_config_file(config_file_path: str) -> None:
    """Export a model using the config file passed."""
    export(get_config_from_file(config_file_path).config)


def export(config: VlmModelConfiguration | LlmModelConfiguration) -> None:
    """Export a model and its components using the passed configuration."""
    model_config = ModelConfigurationInternal(config=config)
    export_conf = model_config.config.export
    if not export_conf.path:
        msg = "Configuration must include a file path at `export.path`."
        raise FileNotFoundError(msg)
    create_directory_if_nonexistant(export_conf.path)
    export_tokenizer = export_conf.tokenizer
    export_processor = export_conf.processor
    export_preprocessor = isinstance(config, VlmModelConfiguration) and export_conf.preprocessor
    _export_model = export_conf.model
    export_config = export_conf.configuration
    setup_logging(model_config.config.log)
    msg = f"Exporting {config.model.name}"
    logger.info(msg)

    # Export tokenizers, processors first. If they have an issue, we should
    if export_tokenizer:
        logger.info("Exporting tokenizer.")
        export_model_tokenizer(model_config)
        logger.info("Tokenizer exported.")
    if export_processor:
        logger.info("Exporting processor.")
        export_model_processor(model_config)
        logger.info("Processor exported.")
    if export_preprocessor:
        logger.info("Exporting preprocessor.")
        export_model_preprocessor(model_config)
        logger.info("Preprocessor exported.")
    if _export_model:
        logger.info("Exporting model.")
        export_model(model_config)
        logger.info("Model exported.")
    if export_config:
        logger.info("Exporting config.")
        dump_config_to_file(config)
        logger.info("Config exported.")


def export_model(model_config: ModelConfigurationInternal) -> None:
    """Convert a model."""
    save_dir = model_config.config.export.path
    model = load_model(model_config)
    assert save_dir
    model.save_pretrained(save_dir)


def export_model_tokenizer(model_config: ModelConfigurationInternal) -> None:
    """Export the model tokenizer."""
    conf = model_config.config
    model_name = conf.model.name
    trust_remote_code = conf.load_options.trust_remote_code
    save_dir = conf.export.path
    hf_tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
    )
    ov_tokenizer, ov_detokenizer = convert_tokenizer(
        hf_tokenizer,
        with_detokenizer=True,
    )
    tokenizer_save_path = f"{save_dir}/{_OV_TOKENIZER_NAME}"
    detokenizer_save_path = f"{save_dir}/{_OV_DETOKENIZER_NAME}"
    save_model(ov_tokenizer, tokenizer_save_path)
    save_model(ov_detokenizer, detokenizer_save_path)


def export_model_processor(model_config: ModelConfigurationInternal) -> None:
    """Export the model processor."""
    conf = model_config.config
    model_name = conf.model.name
    trust_remote_code = conf.load_options.trust_remote_code
    save_dir = conf.export.path
    processor: TokenizersBackend = AutoProcessor.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
    )
    assert save_dir
    processor.save_pretrained(save_dir)


def export_model_preprocessor(model_config: ModelConfigurationInternal) -> None:
    """Export the model pre-processor."""
    conf = model_config.config
    model_name = conf.model.name
    trust_remote_code = conf.load_options.trust_remote_code
    save_dir = conf.export.path
    preprocessor: TokenizersBackend = AutoImageProcessor.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
    )
    assert save_dir
    preprocessor.save_pretrained(save_dir)
