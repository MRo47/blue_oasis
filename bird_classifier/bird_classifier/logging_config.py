import logging
import sys
from logging.handlers import RotatingFileHandler


def setup_logger(
    name: str = "bird_classifier",
    log_file: str | None = None,
    level: int = logging.INFO,
):
    """
    Set up a logger for the bird classifier.

    Parameters
    ----------
    name : str
        The name of the logger. Defaults to "bird_classifier".
    log_file : str | None
        The path to a log file. If None, no file handler is added.
        Defaults to None.
    level : int
        The logging level. Defaults to logging.INFO.

    Returns
    -------
    logger : logging.Logger
        A logger object with the specified name, level, and handlers.
    """
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