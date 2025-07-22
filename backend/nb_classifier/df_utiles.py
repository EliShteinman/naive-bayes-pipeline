import pandas as pd

class DataFrameUtils:
    @staticmethod
    def get_data_as_list_of_dicts(df: pd.DataFrame) -> list[dict]:
        """
        Convert a DataFrame to a list of dictionaries.

        :param df: The DataFrame to convert.
        :return: A list of dictionaries representing the DataFrame rows.
        """
        return df.to_dict(orient='records')