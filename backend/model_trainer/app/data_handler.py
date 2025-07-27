# app/data_handler.py
from typing import Literal

import pandas as pd

from .logger_config import get_logger

logger = get_logger(__name__)

# Define the supported loader types for type hinting and validation
LoaderType = Literal["csv", "tsv", "html"]


class DataHandler:
    """
    Handles loading data from various file formats based on a specified path.
    The loading strategy is determined internally based on the file extension.
    """

    def __init__(self, data_path: str, encoding: str = "utf-8"):
        """
        Initializes the DataHandler with the path to the data file.
        It automatically determines the loader type from the file extension.

        Args:
            data_path (str): The path to the data file to be loaded.
            encoding (str): The file encoding to use. Defaults to 'utf-8'.
        """
        if not data_path:
            raise ValueError("Data path cannot be empty.")

        self.data_path = data_path
        self._encoding = encoding

        # Determine loader type automatically from the file path
        self._loader_type = self._get_loader_type_from_path(data_path)

        # This internal dictionary maps a string identifier to the actual
        # private method responsible for the loading logic.
        self._load_method_map = {
            "csv": self._load_csv,
            "tsv": self._load_tsv,
            "html": self._load_html,
        }

        self._selected_load_method = self._load_method_map[self._loader_type]
        logger.info(
            f"DataHandler initialized for path='{self.data_path}' (type='{self._loader_type}')"
        )

    def load_data(self) -> pd.DataFrame:
        """
        Public method to load data from the configured path.
        This is the single entry point for all loading operations.

        Returns:
            pd.DataFrame: The loaded data.
        """
        logger.info(f"Attempting to load data using method '{self._loader_type}'")
        try:
            return self._selected_load_method()
        except FileNotFoundError:
            logger.error(f"File not found at path: {self.data_path}")
            raise
        except Exception as e:
            logger.error(
                f"Failed to load data from {self.data_path} using method '{self._loader_type}': {e}",
                exc_info=True,
            )
            raise

    def _get_loader_type_from_path(self, path: str) -> LoaderType:
        """Determines the loader type from the file extension."""
        path_lower = path.lower()
        if path_lower.endswith(".csv"):
            return "csv"
        elif path_lower.endswith(".tsv"):
            return "tsv"
        elif path_lower.endswith((".html", ".htm")):
            return "html"
        else:
            # Default to CSV if extension is unknown, or raise an error
            # For robustness, we'll raise an error.
            raise ValueError(f"Could not determine loader type for file: {path}")

    # --- Internal (Private) Implementation Methods ---
    # Note: they no longer need the 'path' argument

    def _load_csv(self) -> pd.DataFrame:
        """Implements the logic for loading a CSV file."""
        logger.debug(f"Executing _load_csv on {self.data_path}")
        return pd.read_csv(self.data_path, encoding=self._encoding)

    def _load_tsv(self) -> pd.DataFrame:
        """Implements the logic for loading a TSV file."""
        logger.debug(f"Executing _load_tsv on {self.data_path}")
        return pd.read_csv(self.data_path, sep="\t", encoding=self._encoding)

    def _load_html(self) -> pd.DataFrame:
        """Implements the logic for loading the first table from an HTML file."""
        logger.debug(f"Executing _load_html on {self.data_path}")
        tables = pd.read_html(self.data_path, encoding=self._encoding, flavor="lxml")
        if not tables:
            raise ValueError(f"No tables found in HTML file: {self.data_path}")
        logger.info(f"Found {len(tables)} tables in HTML, returning the first one.")
        return tables[0]
