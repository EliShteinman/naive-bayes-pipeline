import pandas as pd

from .common.logger_config import get_logger

logger = get_logger(__name__)


class Loader:
    """
    A versatile data loader for reading data from various file formats.

    This class provides a simple interface to load data from sources like
    CSV, JSON, HTML, and Excel files into a pandas DataFrame. It handles
    common errors like file not found or parsing issues.
    """

    def __init__(self, source: str, input_type: str = "csv", encoding: str = "utf-8"):
        """
        Initializes the Loader with the data source and configuration.

        Args:
            source (str): The path or URL to the data file.
            input_type (str, optional): The type of the input file.
                                        Supported types: "csv", "json", "html", "xlsx".
                                        Defaults to "csv".
            encoding (str, optional): The character encoding to use when reading the file.
                                      Defaults to "utf-8".
        """
        self.source = source
        self.input_type = input_type
        self.encoding = encoding
        logger.info(
            f"Loader initialized for source '{self.source}' of type '{self.input_type}'."
        )

    def load(self) -> pd.DataFrame:
        """
        Loads the dataset from the specified source and returns a DataFrame.

        This is the main method to execute the loading process. It dispatches
        to the correct file reader based on the `input_type` and handles
        potential exceptions during the process.

        Returns:
            pd.DataFrame: A DataFrame containing the loaded data.

        Raises:
            FileNotFoundError: If the file specified in `self.source` does not exist.
            ValueError: If the file is empty, cannot be parsed, or the file type
                        is unsupported.
        """
        logger.info(f"Attempting to load data from '{self.source}'...")
        try:
            return self._handle_file_type(self.input_type)
        except FileNotFoundError:
            logger.error(f"File not found: {self.source}")
            raise
        except pd.errors.EmptyDataError:
            logger.error(f"File is empty: {self.source}")
            raise ValueError("The file is empty.")
        except pd.errors.ParserError:
            logger.error(f"Error parsing the file: {self.source}")
            raise ValueError("Error parsing the file.")

    def _handle_file_type(self, file_type: str) -> pd.DataFrame:
        """
        Internal dispatcher to call the appropriate file reading method.

        Args:
            file_type (str): The type of file to read.

        Returns:
            pd.DataFrame: The loaded data.

        Raises:
            ValueError: If the `file_type` is not supported.
        """
        logger.debug(f"Handling file type: '{file_type}'")
        match file_type:
            case "csv":
                return self._read_csv()
            case "json":
                return self._read_json()
            case "html":
                return self._read_html()
            case "xlsx":
                return self._read_excel()
            case _:
                logger.error(f"Unsupported file type: '{file_type}'")
                raise ValueError(f"Unsupported file type: {file_type}")

    def _read_csv(self) -> pd.DataFrame:
        """Reads a CSV file into a DataFrame."""
        logger.info(f"Loading CSV from {self.source} with encoding {self.encoding}")
        df = pd.read_csv(self.source, encoding=self.encoding)
        logger.debug(f"DataFrame shape from CSV: {df.shape}")
        return df

    def _read_json(self) -> pd.DataFrame:
        """Reads a JSON file into a DataFrame."""
        logger.info(f"Loading JSON from {self.source} with encoding {self.encoding}")
        df = pd.read_json(self.source, encoding=self.encoding)
        logger.debug(f"DataFrame shape from JSON: {df.shape}")
        return df

    def _read_html(self) -> pd.DataFrame:
        """
        Reads the first table from an HTML source into a DataFrame.

        Returns:
            pd.DataFrame: The first table found in the HTML content.

        Raises:
            ValueError: If no tables are found in the HTML source.
        """
        logger.info(f"Loading HTML from {self.source}")
        try:
            # pd.read_html returns a list of DataFrames
            tables = pd.read_html(self.source, encoding=self.encoding)
            if not tables:
                # This case is sometimes missed by the exception below
                raise ValueError("No tables found in the HTML source.")
            df = tables[0]
            logger.info(f"Found {len(tables)} table(s) in HTML, loading the first one.")
            logger.debug(f"DataFrame shape from HTML: {df.shape}")
            return df
        except ValueError:
            logger.error(f"No tables found in HTML at: {self.source}")
            raise ValueError("No tables found in the HTML source.")

    def _read_excel(self) -> pd.DataFrame:
        """Reads an Excel file (.xlsx) into a DataFrame."""
        logger.info(f"Loading Excel file from {self.source}")
        # 'openpyxl' is required for reading .xlsx files
        df = pd.read_excel(self.source, engine="openpyxl")
        logger.debug(f"DataFrame shape from Excel: {df.shape}")
        return df
