# backend/nb_classifier/application_manager.py
from typing import Any, Dict, List, Tuple

from .classifier import ClassifierService
from .data_cleaner import DataCleaner
from .data_handler import DataHandler
from .data_splitter import DataSplitter
from .df_utiles import DataFrameUtils
from .logger_config import get_logger
from .model_evaluator import ModelEvaluatorService
from .naive_bayes_model_builder import NaiveBayesModelBuilder

logger = get_logger(__name__)


def display_prediction(sample: Dict, prediction: Any):
    """Utility function to print a prediction result to the console."""
    print(f"\nFor sample: {sample}")
    print(f"The model predicts: {prediction}")


def display_accuracy_report(report: Dict):
    """Utility function to print a formatted model evaluation report."""
    logger.info("--- Model Evaluation Report ---")
    logger.info(f"Accuracy: {report['accuracy']:.2%}")
    logger.info(f"Total samples tested: {report['total_samples']}")
    logger.info(f"Correctly classified: {report['correct_predictions']}")
    logger.info(f"Incorrectly classified: {report['incorrect_predictions']}")
    logger.info("-----------------------------")


def prepare_model_pipeline(
    file_path: str,
    target_col: str,
    min_accuracy: float = 0.8,
) -> Tuple[ClassifierService, Dict[str, List[str]]]:
    """
    Builds the entire model pipeline by orchestrating data loading, cleaning,
    splitting, model building, and evaluation.

    Args:
        file_path (str): Path to the dataset to be processed.
        target_col (str): The name of the target column.
        min_accuracy (float): The minimum required accuracy for the model to be considered valid.

    Returns:
        Tuple[ClassifierService, Dict[str, List[str]]]: A tuple containing the
            ready-to-use ClassifierService and the model's feature schema.

    Raises:
        RuntimeError: If the model's accuracy is below the specified minimum threshold.
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

    # Step 2: Configure and run the data cleaning
    logger.info("Step 2: Cleaning data")
    # The manager's role is to know the specific cleaning configuration for this pipeline
    # and to pass it to the generic DataCleaner.
    data_cleaner = DataCleaner(columns_to_drop=["stalk-root"], remove_constants=True)
    data_cleaned = data_cleaner.clean(data_raw)

    # Step 3: Splitting data
    logger.info(f"Step 3: Splitting data with '{target_col}' as target")
    data_splitter = DataSplitter(data_cleaned, target_col=target_col)
    train_df, test_df = data_splitter.split_data(test_size=0.3, random_state=42)

    # Step 4: Build the Naive Bayes model
    logger.info("Step 4: Building the Naive Bayes model")
    model_builder = NaiveBayesModelBuilder(alpha=1.0)
    trained_model = model_builder.build_model(train_df, target_col=target_col)

    # Step 5: Wrap the model in the ClassifierService
    logger.info("Step 5: Wrapping model in ClassifierService")
    classifier = ClassifierService(model_artifact=trained_model)

    # Step 6: Evaluate performance
    logger.info("Step 6: Evaluating model performance")
    evaluator = ModelEvaluatorService(classifier=classifier)
    list_test_data = DataFrameUtils.get_data_as_list_of_dicts(test_df)
    accuracy_report = evaluator.run_evaluation(
        test_data=list_test_data, target_col=target_col
    )

    display_accuracy_report(accuracy_report)

    # Step 7: Check if accuracy meets the minimum threshold
    if accuracy_report["accuracy"] < min_accuracy:
        msg = (
            f"Model accuracy ({accuracy_report['accuracy']:.2%})"
            f" is below the minimum threshold of {min_accuracy:.2%}. Halting."
        )
        logger.error(msg)
        raise RuntimeError(msg)

    # Step 8: Extract feature schema
    logger.info("Step 8: Extracting feature schema from the model")
    schema = extract_expected_features(trained_model)

    logger.info("Model preparation pipeline completed successfully.")
    return classifier, schema


def extract_expected_features(model: dict) -> Dict[str, List[str]]:
    """
    Extracts the feature schema (features and their possible values) from a trained model artifact.
    This is used to inform the frontend or API clients about the expected input format.
    """
    first_class_dict = next(iter(model.values()))
    return {
        feature: sorted(value_map.keys())
        for feature, value_map in first_class_dict.items()
        if feature != "__prior__"
    }
