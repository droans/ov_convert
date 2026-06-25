"""Control logging settings."""

import logging
import subprocess
from collections.abc import Callable
from io import BufferedReader

from ov_convert.models.misc import LogConfigSchema, LogFilters

logger = logging.getLogger("ov-convert")
logger.addHandler(logging.StreamHandler())


class LogFilter(logging.Filter):
    """Filter logs based on user configuration."""

    def __init__(self, filters: LogFilters) -> None:
        """Set up LogFilter."""
        super().__init__()
        self._filters = filters

    def filter(self, record: logging.LogRecord) -> bool:
        """Determine if log should be passed."""
        if record.name in self._filters.components:
            return False
        try:
            return not any(item in record.message for item in self._filters.messages)
        except:  # noqa: E722
            return not any(item in record.msg for item in self._filters.messages)


def setup_logging(log_config: LogConfigSchema) -> None:
    """Set the logging settings based on the configuration."""
    handler = (
        logging.FileHandler(log_config.path, "w+") if log_config.path else logging.StreamHandler()
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


def _log_popen(pipe: BufferedReader, log_fn: Callable) -> str:
    result = ""
    for line in iter(pipe.readline, b""):
        tmp = line.decode()
        log_fn(tmp.strip())
        result += tmp
    return result


def logged_popen(
    log_fn: Callable = logger.debug,
    *args,  # noqa: ANN002
    log_stdin: bool = False,
    log_stdout: bool = True,
    log_stderr: bool = True,
    shell: bool = False,
    cwd: str | None = None,
    **kwargs,  # noqa: ANN003
) -> tuple[subprocess.Popen, str]:
    """Run subprocess.popen and send output to logs."""
    if log_stdin:
        kwargs["stdin"] = subprocess.STDOUT
    if log_stdout:
        kwargs["stdout"] = subprocess.PIPE
    if log_stderr:
        kwargs["stderr"] = subprocess.STDOUT
    process = subprocess.Popen(  # noqa: S603
        *args,
        shell=shell,
        cwd=cwd,
        **kwargs,
    )
    result = ""
    if isinstance(process.stdout, BufferedReader):
        with process.stdout:
            result = _log_popen(process.stdout, log_fn)
    else:
        log_fn("No stdout - not logging.")
    process.kill()
    return (process, result)
