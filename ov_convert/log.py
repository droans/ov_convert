"""Control logging settings."""

import logging

from ov_convert.models.misc import LogConfigSchema, LogFilters

logger = logging.getLogger("ov-convert")
logger.addHandler(logging.StreamHandler())


class LogFilter(logging.Filter):
    """Filter logs based on user configuration."""

    def __init__(self, filters: LogFilters) -> None:
        """Setup LogFilter."""
        super().__init__()
        self._filters = filters

    def filter(self, record: logging.LogRecord) -> bool:
        """Determine if log should be passed."""
        if record.name in self._filters.components:
            return False
        return not any(item in record.message for item in self._filters.messages)


def setup_logging(log_config: LogConfigSchema) -> None:
    """Sets the logging settings based on the configuration."""
    handler = (
        logging.FileHandler(log_config.path, "w+")
        if log_config.path
        else logging.StreamHandler()
    )
    handler.addFilter(LogFilter(log_config.filters))
    handler.setFormatter(logging.Formatter(log_config.format))
    handler.setLevel(log_config.level.upper())
    logging.basicConfig(
        format=log_config.format,
        level=log_config.level.upper(),
        handlers=[handler],
    )
    logger.addFilter(LogFilter(log_config.filters))
    logging.root.addHandler(handler)
