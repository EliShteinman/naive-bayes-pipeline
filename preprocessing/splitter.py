# preprocessing/splitter.py
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple
from config.logger_config import get_logger

logger = get_logger(__name__)


class Splitter:
    def __init__(
        self,
        data: pd.DataFrame,
        target_col: str,
        test_size: float = 0.3,
        random_state: int = 42,
        stratify: bool = True,
    ):
        self.data = data
        self.target_col = target_col
        self.test_size = test_size
        self.random_state = random_state
        self.stratify = stratify

    def split(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        stratify_values = self.data[self.target_col] if self.stratify else None





import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple


class Splitter:
    def __init__(
        self,
        data: pd.DataFrame,
        target_col: str,
        test_size: float = 0.3,
        random_state: int = 42,
        stratify: bool = True,
    ):
        self.data = data
        self.target_col = target_col
        self.test_size = test_size
        self.random_state = random_state
        self.stratify = stratify

    def split(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        # מבצעים את הפיצול רגיל, כולל stratify
        stratify_values = self.data[self.target_col] if self.stratify else None

        train_df, test_df = train_test_split(
            self.data,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=stratify_values,
        )

        # בדיקה אמיתית: האם יש מחלקות בטסט שלא קיימות באימון?
        test_classes = set(test_df[self.target_col])
        train_classes = set(train_df[self.target_col])
        missing = test_classes - train_classes

        if missing:
            # העבר את כל השורות ששייכות למחלקות החסרות לטסט → ל־train
            move_back_mask = test_df[self.target_col].isin(missing)
            move_back = test_df[move_back_mask]
            test_df = test_df[~move_back_mask]
            train_df = pd.concat([train_df, move_back], ignore_index=True)

            print(f"⚠️ זיהינו מחלקות שלא היו באימון: {missing}. העברנו אותן חזרה ל־train.")

        return train_df, test_df, pd.DataFrame(columns=self.data.columns)  # אין dropped אמיתי כאן
