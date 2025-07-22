# backend/nb_classifier/df_utiles.py
import pandas as pd
from typing import List, Dict, Any


class DataFrameUtils:
    """
    A utility class for common DataFrame operations.
    """

    @staticmethod
    def get_data_as_list_of_dicts(df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Convert a pandas DataFrame to a list of dictionaries.

        Each dictionary in the list represents a row from the DataFrame,
        with column names as keys.

        Args:
            df (pd.DataFrame): The DataFrame to convert.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame.")
        return df.to_dict(orient="records")
