import logging
import os
import sys

# Constants for formatting
CONSOLE_FORMAT = "%(asctime)s - %(name)-20s - %(levelname)-8s - %(message)s"
FILE_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s"
LOG_FILE = "backend_app.log"

# A flag to ensure setup happens only once
_logging_configured = False


def setup_logging():
    """
    Configures the root logger for the application.
    This function should be called only once when the application starts.
    """
    global _logging_configured
    if _logging_configured:
        return

    # Read configuration from environment variables at runtime
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    log_to_file = os.getenv("LOG_TO_FILE", "false").lower() == "true"

    # Get the root logger
    root_logger = logging.getLogger()
    # Clear any existing handlers to avoid duplicates
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    # Set the base level to capture all messages; handlers will filter them
    root_logger.setLevel(logging.DEBUG)

    # --- Console Handler ---
    console_handler = logging.StreamHandler(sys.stdout)
    try:
        console_handler.setLevel(log_level)
    except ValueError:
        # Fallback to INFO if an invalid level is provided
        console_handler.setLevel(logging.INFO)
        logging.warning(f"Invalid LOG_LEVEL '{log_level}'. Defaulting to INFO.")

    console_handler.setFormatter(logging.Formatter(CONSOLE_FORMAT, datefmt="%H:%M:%S"))
    root_logger.addHandler(console_handler)

    # --- File Handler (Conditional) ---
    if log_to_file:
        try:
            file_handler = logging.FileHandler(LOG_FILE, mode="a")
            file_handler.setLevel(logging.WARNING)  # Log only important messages to file
            file_handler.setFormatter(logging.Formatter(FILE_FORMAT))
            root_logger.addHandler(file_handler)
            logging.info(f"Logging to file enabled: {LOG_FILE}")
        except (IOError, PermissionError) as e:
            logging.warning(f"Could not configure file logging to {LOG_FILE}. Error: {e}")
    else:
        # This message will now be logged once by the configured logger
        logging.info("Logging to file is disabled.")

    _logging_configured = True


def get_logger(name: str) -> logging.Logger:
    """
    Returns a logger instance with the specified name.
    Assumes setup_logging() has already been called.
    """
    return logging.getLogger(name)