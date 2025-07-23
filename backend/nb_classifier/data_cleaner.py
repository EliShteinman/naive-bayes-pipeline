# backend/nb_classifier/data_cleaner.py
import pandas as pd
from typing import List, Callable
from .logger_config import get_logger

logger = get_logger(__name__)

# A "Cleaning Step" is any function that accepts a DataFrame and returns a DataFrame.
# This is a powerful concept using Python's first-class functions.
CleaningStep = Callable[[pd.DataFrame], pd.DataFrame]


class DataCleaner:
    """
    Executes a sequence of cleaning steps on a DataFrame.
    This allows for a flexible and configurable cleaning pipeline, where the
    specific cleaning logic is injected from the outside.
    """

    def __init__(self, cleaning_steps: List[CleaningStep]):
        """
        Initializes the DataCleaner with a list of cleaning steps to be executed.

        Args:
            cleaning_steps (List[CleaningStep]): A list of functions, where each
                function performs one cleaning operation on a DataFrame.
        """
        self.steps = cleaning_steps
        logger.info(f"DataCleaner initialized with {len(self.steps)} cleaning steps.")

    def clean(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Applies all configured cleaning steps sequentially to the DataFrame.

        Args:
            data (pd.DataFrame): The raw DataFrame to be cleaned.

        Returns:
            pd.DataFrame: The cleaned DataFrame.
        """
        logger.info("Starting data cleaning process...")
        # We work on a copy to avoid side effects on the original DataFrame
        cleaned_data = data.copy()

        for i, step in enumerate(self.steps):
            # Get the function name for clearer logging
            step_name = getattr(step, '__name__', 'custom_step')
            if hasattr(step, 'func'):  # Handles functools.partial objects
                step_name = f"partial({step.func.__name__})"

            logger.debug(f"Executing cleaning step {i + 1}/{len(self.steps)}: {step_name}")
            cleaned_data = step(cleaned_data)

        logger.info("Data cleaning finished.")
        return cleaned_data