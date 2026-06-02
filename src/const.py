"""Constants."""
import typing
from typing import Literal

QUANT_BACKUP_PRECISION_TYPE = Literal["int8_sym", "int8_asym"]
QUANT_GROUP_SIZE_FALLBACK_TYPE = Literal["error", "ignore", "adjust"]
QUANT_SENSITIVITIY_METRIC_TYPE = Literal[
  "weight_quantization_error",
  "hessian_input_activation",
  "mean_activation_variance",
  "max_activation_variance",
  "mean_activation_magnitude",
]
QUANT_TASKS_TYPES_TYPE = Literal[
  "audio-classification",
  "audio-frame-classification",
  "audio-xvector",
  "automatic-speech-recognition",
  "automatic-speech-recognition-with-past",
  "depth-estimation",
  "document-question-answering",
  "document-question-answering-with-past",
  "feature-extraction",
  "feature-extraction-with-past",
  "fill-mask",
  "image-classification",
  "image-segmentation",
  "image-to-image",
  "image-to-text",
  "image-to-text-with-past",
  "inpainting",
  "mask-generation",
  "masked-im",
  "multiple-choice",
  "object-detection",
  "question-answering",
  "semantic-segmentation",
  "sentence-similarity",
  "text-classification",
  "text-generation",
  "text-generation-with-past",
  "text-to-audio",
  "text-to-audio-with-past",
  "text-to-image",
  "text2text-generation",
  "text2text-generation-with-past",
  "token-classification",
  "zero-shot-image-classification",
  "zero-shot-object-detection",
]

QUANT_BACKUP_PRECISIONS = list(typing.get_args(QUANT_BACKUP_PRECISION_TYPE))
QUANT_GROUP_SIZE_FALLBACKS = list(typing.get_args(QUANT_GROUP_SIZE_FALLBACK_TYPE))
QUANT_SENSITIVITIY_METRICS = list(typing.get_args(QUANT_SENSITIVITIY_METRIC_TYPE))
QUANT_TASKS_TYPE = list(typing.get_args(QUANT_TASKS_TYPES_TYPE))

QUANT_DEFAULT_BACKUP_PRECISION: QUANT_BACKUP_PRECISION_TYPE = "int8_asym"
QUANT_DEFAULT_GROUP_SIZE_FALLBACK: QUANT_GROUP_SIZE_FALLBACK_TYPE = "adjust"
QUANT_DEFAULT_SENSITIVITIY_METRIC: QUANT_SENSITIVITIY_METRIC_TYPE = "weight_quantization_error"
