# logger_config.py
import logging

logging.basicConfig(
    filename="myapp.log",
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def get_logger(name):
    return logging.getLogger(name)
