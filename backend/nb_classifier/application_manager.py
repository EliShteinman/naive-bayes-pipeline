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

# --- Constants ---
FILE_PATH = "data/mushroom_decoded.csv"
TARGET_COL = "poisonous"


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
    file_path: str = FILE_PATH,
    target_col: str = TARGET_COL,
    min_accuracy: float = 0.8,
) -> Tuple[ClassifierService, Dict[str, List[str]]]:
    """
    Builds the entire model pipeline: loads data, trains, evaluates, and
    returns the classifier service and the feature schema.

    Args:
        file_path (str): Path to the dataset CSV file.
        target_col (str): The name of the target column.
        min_accuracy (float): The minimum required accuracy for the model to be considered valid.

    Returns:
        Tuple[ClassifierService, Dict[str, List[str]]]: A tuple containing the
            ready-to-use ClassifierService and the model's feature schema.

    Raises:
        RuntimeError: If the model's accuracy is below the specified minimum threshold.
    """
    logger.info("Starting full model preparation pipeline...")

    # 1. Load, clean, and split the data
    logger.info(f"Step 1: Loading data from '{file_path}'")
    data_handler = DataHandler(data_path=file_path)
    data_raw = data_handler.load_data()

    logger.info("Step 2: Cleaning data")
    data_cleaner = DataCleaner(data_raw)
    data_cleaned = data_cleaner.clean()

    logger.info(f"Step 3: Splitting data with '{target_col}' as target")
    data_splitter = DataSplitter(data_cleaned, target_col=target_col)
    train_df, test_df = data_splitter.split_data(test_size=0.3, random_state=42)

    # 2. Build the Naive Bayes model
    logger.info("Step 4: Building the Naive Bayes model")
    model_builder = NaiveBayesModelBuilder(alpha=1.0)
    trained_model = model_builder.build_model(train_df, target_col=target_col)

    # 3. Wrap the model in the ClassifierService
    logger.info("Step 5: Wrapping model in ClassifierService")
    classifier = ClassifierService(model_artifact=trained_model)

    # 4. Evaluate performance
    logger.info("Step 6: Evaluating model performance")
    evaluator = ModelEvaluatorService(classifier=classifier)
    list_test_data = DataFrameUtils.get_data_as_list_of_dicts(test_df)
    accuracy_report = evaluator.run_evaluation(
        test_data=list_test_data, target_col=target_col
    )

    # Display the results using the utility function
    display_accuracy_report(accuracy_report)

    # 5. Check if accuracy meets the minimum threshold
    if accuracy_report["accuracy"] < min_accuracy:
        msg = f"Model accuracy ({accuracy_report['accuracy']:.2%})"\
              f" is below the minimum threshold of {min_accuracy:.2%}. Halting."
        logger.error(msg)
        raise RuntimeError(msg)

    # 6. Display a sample prediction
    logger.info("Step 7: Extracting feature schema from the model")
    schema = extract_expected_features(trained_model)

    logger.info("Model preparation pipeline completed successfully.")
    return classifier, schema


def extract_expected_features(model: dict) -> Dict[str, List[str]]:
    """
    Extracts the feature schema (features and their possible values) from a trained model artifact.

    This is used to inform the frontend or API clients about the expected input format.

    Args:
        model (dict): The trained model artifact.

    Returns:
        Dict[str, List[str]]: A dictionary where keys are feature names and values are sorted
                              lists of their possible values.
    """
    # Get the first class dictionary from the model (e.g., 'poisonous')
    first_class_dict = next(iter(model.values()))

    # Extract features and their value maps, excluding the internal '__prior__' key
    return {
        feature: sorted(value_map.keys())
        for feature, value_map in first_class_dict.items()
        if feature != "__prior__"
    }
