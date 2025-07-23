# backend/nb_classifier/application_manager.py
from typing import Any, Dict, List, Tuple

from .classifier import ClassifierService
from .data_cleaner import DataCleaner
from .data_handler import DataHandler
from .data_splitter import DataSplitter
from .df_utiles import DataFrameUtils
from .logger_config import get_logger
from .model_evaluator import ModelEvaluatorService
# The builder now returns an IModelArtifact, but the manager doesn't need to know the specific type.
from .naive_bayes_model_builder import NaiveBayesModelBuilder

logger = get_logger(__name__)


def display_prediction(sample: Dict, prediction: Any):
    """Utility function to print a prediction result to the console."""
    print(f"\nFor sample: {sample}")
    print(f"The model predicts: {prediction}")


def display_accuracy_report(report: Dict, pos_label: Any):
    """Utility function to print a formatted model evaluation report."""
    logger.info("--- Model Evaluation Report ---")
    logger.info(f"Positive Label for Metrics: '{pos_label}'")
    logger.info(f"Accuracy: {report['accuracy']:.2%}")
    logger.info(f"Precision: {report['precision']:.2f}")
    logger.info(f"Recall: {report['recall']:.2f}")
    logger.info(f"F1-Score: {report['f1_score']:.2f}")
    logger.info(f"Total samples tested: {report['total_samples']}")
    logger.info(f"Correctly classified: {report['correct_predictions']}")
    logger.info(f"Incorrectly classified: {report['incorrect_predictions']}")
    logger.info("-----------------------------")


def prepare_model_pipeline(
    file_path: str,
    target_col: str,
    pos_label: Any,
    min_accuracy: float = 0.8,
) -> Tuple[ClassifierService, Dict[str, List[str]]]:
    """
    Builds the entire model pipeline and returns the classifier and its schema.

    Args:
        file_path (str): Path to the dataset to be processed.
        target_col (str): The name of the target column.
        pos_label (Any): The value in the target column considered "positive".
        min_accuracy (float): The minimum required accuracy for the model.

    Returns:
        A tuple containing the ready-to-use ClassifierService and the model's schema.
    """
    logger.info("Starting full model preparation pipeline...")

    # Step 1: Load data
    logger.info(f"Step 1: Preparing to load data from '{file_path}'")
    data_handler = DataHandler(data_path=file_path)
    data_raw = data_handler.load_data()

    # Step 2: Clean data
    logger.info("Step 2: Cleaning data")
    data_cleaner = DataCleaner(
        columns_to_drop=['stalk-root'],
        remove_constants=True
    )
    data_cleaned = data_cleaner.clean(data_raw)

    # Step 3: Split data
    logger.info(f"Step 3: Splitting data with '{target_col}' as target")
    data_splitter = DataSplitter(data_cleaned, target_col=target_col, test_size=0.3, random_state=42)
    train_df, test_df = data_splitter.split_data()

    # Step 4: Build the model artifact
    logger.info("Step 4: Building the Naive Bayes model artifact")
    model_builder = NaiveBayesModelBuilder(alpha=1.0)
    # This now returns an IModelArtifact object, not a raw dict
    trained_model_artifact = model_builder.build_model(train_df, target_col=target_col)

    # Step 5: Wrap the model artifact in the ClassifierService
    logger.info("Step 5: Wrapping model artifact in ClassifierService")
    classifier = ClassifierService(model_artifact=trained_model_artifact)

    # Step 6: Evaluate performance
    logger.info("Step 6: Evaluating model performance")
    evaluator = ModelEvaluatorService(classifier=classifier)
    list_test_data = DataFrameUtils.get_data_as_list_of_dicts(test_df)
    accuracy_report = evaluator.run_evaluation(
        test_data=list_test_data, target_col=target_col, pos_label=pos_label
    )

    display_accuracy_report(accuracy_report, pos_label=pos_label)

    # Step 7: Check if accuracy meets the minimum threshold
    if accuracy_report["accuracy"] < min_accuracy:
        msg = f"Model accuracy ({accuracy_report['accuracy']:.2%}) is below the minimum threshold of {min_accuracy:.2%}. Halting."
        logger.error(msg)
        raise RuntimeError(msg)

    # Step 8: Extract feature schema directly from the artifact
    logger.info("Step 8: Extracting feature schema from the model artifact")
    # CHANGED: The artifact itself knows how to provide its schema.
    schema = trained_model_artifact.get_schema()

    logger.info("Model preparation pipeline completed successfully.")
    return classifier, schema