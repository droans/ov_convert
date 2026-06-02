"""Constants."""
import typing
from typing import Literal

BACKUP_PRECISION_TYPE = Literal["int8_sym", "int8_asym"]
GROUP_SIZE_FALLBACK_TYPE = Literal["error", "ignore", "adjust"]
SENSITIVITIY_METRIC_TYPE = Literal[
  "weight_quantization_error",
  "hessian_input_activation",
  "mean_activation_variance",
  "max_activation_variance",
  "mean_activation_magnitude",
]
TASKS_TYPES_TYPE = Literal[
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

BACKUP_PRECISIONS = list(typing.get_args(BACKUP_PRECISION_TYPE))
GROUP_SIZE_FALLBACKS = list(typing.get_args(GROUP_SIZE_FALLBACK_TYPE))
SENSITIVITIY_METRICS = list(typing.get_args(SENSITIVITIY_METRIC_TYPE))
TASKS_TYPE = list(typing.get_args(TASKS_TYPES_TYPE))

DEFAULT_BACKUP_PRECISION: BACKUP_PRECISION_TYPE = "int8_asym"
DEFAULT_GROUP_SIZE_FALLBACK: GROUP_SIZE_FALLBACK_TYPE = "adjust"
