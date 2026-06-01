"""Constants."""

QUANT_OPTIONS_BACKUP_PRECISION: list[str | None] = [None, "int8_sym", "int8_asym"]
QUANT_OPTIONS_GROUP_SIZE_FALLBACK: list[str | None] = [None, "error", "ignore", "adjust"]
QUANT_OPTIONS_SENSITIVITIY_METRIC: list[str | None] = [
  None,
  "weight_quantization_error",
  "hessian_input_activation",
  "mean_activation_variance",
  "max_activation_variance",
  "mean_activation_magnitude",
]

QUANT_DEFAULT_BACKUP_PRECISION = "int8_asym"
QUANT_DEFAULT_GROUP_SIZE_FALLBACK = "adjust"
QUANT_DEFAULT_SENSITIVITIY_METRIC = "weight_quantization_error"
