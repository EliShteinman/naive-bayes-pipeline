# backend/nb_classifier/logger_config.py
import logging
import os
import sys

# --- Configuration from Environment Variables ---
# LOG_TO_FILE controls whether we write to a file. Defaults to 'true' for local dev.
# In a Docker environment, this should be set to 'false' or any other value.
LOG_TO_FILE = os.getenv("LOG_TO_FILE", "true").lower() == "true"
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

# --- Formats and Constants ---
CONSOLE_FORMAT = "%(asctime)s - %(name)-20s - %(levelname)-8s - %(message)s"
FILE_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s"
LOG_FILE = "backend_app.log"


def get_logger(name: str) -> logging.Logger:
    """
    Configures and returns a logger.
    - Always logs to the console.
    - Conditionally logs to a file based on the LOG_TO_FILE environment variable.
    """
    logger = logging.getLogger(name)

    # Set the level for the logger itself. Messages below this will be ignored.
    logger.setLevel(logging.DEBUG)  # Process everything, handlers will filter

    if not logger.handlers:
        # --- Console Handler (Always On) ---
        console_handler = logging.StreamHandler(sys.stdout)

        # Set level from environment variable, fallback to INFO
        try:
            console_handler.setLevel(LOG_LEVEL)
        except ValueError:
            print(f"Warning: Invalid LOG_LEVEL '{LOG_LEVEL}'. Defaulting to INFO.")
            console_handler.setLevel(logging.INFO)

        console_handler.setFormatter(logging.Formatter(CONSOLE_FORMAT, datefmt="%H:%M:%S"))
        logger.addHandler(console_handler)

        # --- File Handler (Conditional) ---
        if LOG_TO_FILE:
            try:
                # Writes logs to a file. Mode 'a' appends to the file.
                file_handler = logging.FileHandler(LOG_FILE, mode='a')
                file_handler.setLevel(logging.WARNING)  # Usually, we want only important logs in files
                file_handler.setFormatter(logging.Formatter(FILE_FORMAT))
                logger.addHandler(file_handler)
                # Use a one-time print to inform the user, not a logger call
                print(f"Logging to file enabled: {LOG_FILE}")
            except (IOError, PermissionError) as e:
                # A one-time print is better here, as logger might be in a broken state
                print(f"Warning: Could not configure file logging to {LOG_FILE}. Error: {e}")
        else:
            print("Logging to file is disabled.")

    return logger