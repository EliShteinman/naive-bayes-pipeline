# nb_classifier/data_splitter.py
import pandas as pd
from typing import Tuple
from .logger_config import get_logger
from sklearn.model_selection import train_test_split

logger = get_logger(__name__)

class DataSplitter:
    def __init__(self, data: pd.DataFrame, target_col: str):
        self.data = data
        self.target_col = target_col

    def split_data(self, test_size: float = 0.3, random_state: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
        train_df, test_df = self._split_and_validate_df(
            test_size=test_size,
            random_state=random_state,
        )
        return train_df, test_df

    def _split_and_validate_df(
        self,
        test_size: float = 0.3,
        random_state: int = 42,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:

        train_df, test_df = train_test_split(
            self.data,
            test_size=test_size,
            random_state=random_state,
            stratify=self.data[self.target_col],
        )

        feature_cols = [col for col in self.data.columns if col != self.target_col]
        train_uniques = {col: set(train_df[col].unique()) for col in feature_cols}

        mask_to_drop = (
            ~test_df[feature_cols]
            .apply(lambda col: col.isin(train_uniques[col.name]))
            .all(axis=1)
        )

        indices_to_drop = test_df[mask_to_drop].index

        if not indices_to_drop.empty:
            logger.info(
                f"הוסרו {len(indices_to_drop)} שורות מסט הבדיקה בגלל ערכים שלא הופיעו באימון."
            )
            test_df = test_df.drop(index=indices_to_drop)

        return train_df.reset_index(drop=True), test_df.reset_index(drop=True)