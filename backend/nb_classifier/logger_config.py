# backend/nb_classifier/logger_config.py
import logging
import sys

# Define different formats for console and file output
CONSOLE_FORMAT = "%(asctime)s - %(name)-20s - %(levelname)-8s - %(message)s"
FILE_FORMAT = (
    "%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s"
)
LOG_FILE = "backend_app.log"


def get_logger(name: str) -> logging.Logger:
    """
    Configures and returns a logger with handlers for console and file.

    The configuration is applied only once to the root logger to avoid
    duplicate handlers.

    - Console Handler: Logs INFO level and above.
    - File Handler: Logs WARNING level and above.

    Args:
        name (str): The name for the logger, typically __name__.

    Returns:
        logging.Logger: A configured logger instance.
    """
    logger = logging.getLogger(name)

    # Set the lowest level to be processed by the logger.
    logger.setLevel(logging.DEBUG)

    # Prevent adding duplicate handlers if the function is called again.
    if not logger.handlers:
        # --- Console Handler ---
        # Prints logs to the standard output.
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(
            logging.Formatter(CONSOLE_FORMAT, datefmt="%H:%M:%S")
        )
        logger.addHandler(console_handler)

        # --- File Handler ---
        # Writes logs to a file. Mode 'a' appends to the file.
        file_handler = logging.FileHandler(LOG_FILE, mode="a")
        file_handler.setLevel(logging.WARNING)
        file_handler.setFormatter(logging.Formatter(FILE_FORMAT))
        logger.addHandler(file_handler)

    return logging.getLogger(name)
