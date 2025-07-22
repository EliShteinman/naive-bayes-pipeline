# nb_classifier/data_cleaner.py
import pandas as pd

class DataCleaner:
    def __init__(self, data: pd.DataFrame):
        self.data = data

    def clean(self) -> pd.DataFrame:
        return self._load_and_clean_df()

    def _load_and_clean_df(self) -> pd.DataFrame:
        cleaned_data = (self.data.drop(columns=["stalk-root"], errors="ignore").loc[:, lambda df: df.nunique() > 1])
        return cleaned_data