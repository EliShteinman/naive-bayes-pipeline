# service_evaluator/app/common/logger_config.py
import logging
import os
import sys


def setup_logger():
    """
    Sets up a centralized logger that writes to both a file and the console.

    The log level can be controlled via the 'LOG_LEVEL' environment variable.
    Defaults to INFO.
    """
    # Get log level from environment variable, default to INFO
    log_level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, log_level_name, logging.INFO)

    # Create logger
    logger = logging.getLogger("nb_classifier")  # Get the root logger for your app
    logger.setLevel(log_level)

    # Avoid adding handlers multiple times if this function is called more than once
    if logger.hasHandlers():
        logger.handlers.clear()

    # --- Formatter ---
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # --- Console Handler (stdout) ---
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(log_level)
    stdout_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler)

    # --- File Handler ---
    # It's good practice to ensure the directory exists
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    file_handler = logging.FileHandler(
        f"{log_dir}/service_evaluator.log", encoding="utf-8"
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


# A single function to get a configured logger
def get_logger(name: str) -> logging.Logger:
    """Returns a logger instance for a specific module."""
    # This will get a child logger of the root 'nb_classifier' logger
    return logging.getLogger(f"nb_classifier.{name}")
