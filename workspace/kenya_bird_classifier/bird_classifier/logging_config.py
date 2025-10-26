import logging
import sys
from logging.handlers import RotatingFileHandler


def setup_logger(
    name: str = "bird_classifier",
    log_file: str | None = None,
    level: int = logging.INFO,
):
    """Configure and return a logger with console + optional file output."""

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    # avoid adding duplicate handlers
    if logger.hasHandlers():
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file:
        file_handler = RotatingFileHandler(log_file, maxBytes=5_000_000, backupCount=5)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

logger = setup_logger()