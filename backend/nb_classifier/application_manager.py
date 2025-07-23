# backend/nb_classifier/application_manager.py
from typing import Any, Dict, List, Tuple
from functools import partial
import pandas as pd

from .classifier import ClassifierService
from .data_cleaner import DataCleaner
from .data_handler import DataHandler
from .data_splitter import DataSplitter
from .df_utiles import DataFrameUtils
from .logger_config import get_logger
from .model_evaluator import ModelEvaluatorService
from .naive_bayes_model_builder import NaiveBayesModelBuilder

logger = get_logger(__name__)


# --- Reusable, Generic Cleaning Functions ---
# These functions define individual, reusable cleaning operations.

def drop_specified_columns(df: pd.DataFrame, columns_to_drop: List[str]) -> pd.DataFrame:
    """A generic cleaning step to drop a list of columns."""
    logger.info(f"Attempting to drop columns: {columns_to_drop}")
    return df.drop(columns=columns_to_drop, errors="ignore")


def remove_columns_with_single_unique_value(df: pd.DataFrame) -> pd.DataFrame:
    """A generic cleaning step to remove columns that have no variance (constant columns)."""
    cols_before = set(df.columns)
    # The .loc[:, df.nunique() > 1] syntax is a concise way to select columns
    # where the number of unique values is greater than 1.
    cleaned_df = df.loc[:, df.nunique() > 1]
    cols_after = set(cleaned_df.columns)

    removed_cols = cols_before - cols_after
    if removed_cols:
        logger.info(f"Removed constant columns with no variance: {sorted(list(removed_cols))}")
    return cleaned_df


# --- Utility Functions (display_... functions are unchanged) ---

def display_prediction(sample: Dict, prediction: Any):
    # ... (code unchanged)
    print(f"\nFor sample: {sample}")
    print(f"The model predicts: {prediction}")


def display_accuracy_report(report: Dict):
    # ... (code unchanged)
    logger.info("--- Model Evaluation Report ---")
    logger.info(f"Accuracy: {report['accuracy']:.2%}")
    logger.info(f"Total samples tested: {report['total_samples']}")
    logger.info(f"Correctly classified: {report['correct_predictions']}")
    logger.info(f"Incorrectly classified: {report['incorrect_predictions']}")
    logger.info("-----------------------------")


# --- Main Pipeline Orchestrator ---

def prepare_model_pipeline(
        file_path: str,
        target_col: str,
        min_accuracy: float = 0.8,
) -> Tuple[ClassifierService, Dict[str, List[str]]]:
    """
    Builds the entire model pipeline by orchestrating data loading, cleaning,
    splitting, model building, and evaluation.
    ... (rest of docstring) ...
    """
    logger.info("Starting full model preparation pipeline...")

    # Step 1: Load data using the self-sufficient DataHandler
    logger.info(f"Step 1: Preparing to load data from '{file_path}'")
    try:
        data_handler = DataHandler(data_path=file_path)
        data_raw = data_handler.load_data()
    except (ValueError, FileNotFoundError) as e:
        logger.error(f"Data loading failed: {e}")
        raise

    # Step 2: Configure and run the data cleaning pipeline
    logger.info("Step 2: Building and running the cleaning pipeline")

    # Define the specific "recipe" of cleaning steps for the mushroom dataset.
    # This makes the cleaning process explicit and configurable.
    mushroom_cleaning_recipe = [
        # Create a specialized function on-the-fly to drop 'stalk-root'
        partial(drop_specified_columns, columns_to_drop=['stalk-root']),
        # The second step is a direct reference to the function
        remove_columns_with_single_unique_value
    ]

    # Initialize the generic cleaner with our specific recipe
    data_cleaner = DataCleaner(cleaning_steps=mushroom_cleaning_recipe)
    data_cleaned = data_cleaner.clean(data_raw)

    # --- The rest of the pipeline continues as before ---

    logger.info(f"Step 3: Splitting data with '{target_col}' as target")
    data_splitter = DataSplitter(data_cleaned, target_col=target_col)
    train_df, test_df = data_splitter.split_data(test_size=0.3, random_state=42)

    logger.info("Step 4: Building the Naive Bayes model")
    model_builder = NaiveBayesModelBuilder(alpha=1.0)
    trained_model = model_builder.build_model(train_df, target_col=target_col)

    logger.info("Step 5: Wrapping model in ClassifierService")
    classifier = ClassifierService(model_artifact=trained_model)

    logger.info("Step 6: Evaluating model performance")
    evaluator = ModelEvaluatorService(classifier=classifier)
    list_test_data = DataFrameUtils.get_data_as_list_of_dicts(test_df)
    accuracy_report = evaluator.run_evaluation(
        test_data=list_test_data, target_col=target_col
    )

    display_accuracy_report(accuracy_report)

    if accuracy_report["accuracy"] < min_accuracy:
        msg = f"Model accuracy ({accuracy_report['accuracy']:.2%}) is below the minimum threshold of {min_accuracy:.2%}. Halting."
        logger.error(msg)
        raise RuntimeError(msg)

    logger.info("Step 7: Extracting feature schema from the model")
    schema = extract_expected_features(trained_model)

    logger.info("Model preparation pipeline completed successfully.")
    return classifier, schema


def extract_expected_features(model: dict) -> Dict[str, List[str]]:
    # ... (code unchanged)
    first_class_dict = next(iter(model.values()))
    return {
        feature: sorted(value_map.keys())
        for feature, value_map in first_class_dict.items()
        if feature != "__prior__"
    }