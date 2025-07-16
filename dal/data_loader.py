# dal/data_loader.py
import pandas as pd
from config.logger_config import get_logger

logger = get_logger(__name__)


class Loader:
    def __init__(self, source: str, input_type: str = "csv", encoding: str = "utf-8"):
        self.source = source
        self.input_type = input_type
        self.encoding = encoding

    def load(self) -> pd.DataFrame:
        """
        Load the dataset from the specified source and return a DataFrame.
        """
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
                raise ValueError(f"Unsupported file type: {file_type}")

    def _read_csv(self) -> pd.DataFrame:
        logger.info(f"Loading CSV from {self.source} with encoding {self.encoding}")
        df = pd.read_csv(self.source, encoding=self.encoding)
        logger.debug(f"DataFrame shape: {df.shape}")
        return df

    def _read_json(self) -> pd.DataFrame:
        logger.info(f"Loading JSON from {self.source} with encoding {self.encoding}")
        df = pd.read_json(self.source, encoding=self.encoding)
        logger.debug(f"DataFrame shape: {df.shape}")
        return df

    def _read_html(self) -> pd.DataFrame:
        logger.info(f"Loading HTML from {self.source}")
        try:
            tables = pd.read_html(self.source, encoding=self.encoding)
            df = tables[0]
            logger.debug(f"DataFrame shape: {df.shape}")
            return df
        except ValueError:
            logger.error(f"No tables found in HTML at: {self.source}")
            raise ValueError("No tables found in the HTML source.")

    def _read_excel(self) -> pd.DataFrame:
        logger.info(f"Loading Excel file from {self.source}")
        df = pd.read_excel(self.source, engine="openpyxl")
        logger.debug(f"DataFrame shape: {df.shape}")
        return df
