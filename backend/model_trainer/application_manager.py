# application_manager.py
from typing import Any, Dict

from app.data_cleaner import DataCleaner
from app.data_handler import DataHandler
from app.data_splitter import DataSplitter
from app.logger_config import get_logger
from app.model_evaluator import ModelEvaluatorService
from app.naive_bayes_model_builder import NaiveBayesModelBuilder
from app.model_artifact import IModelArtifact

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
) -> IModelArtifact:
    """
    Builds the entire model pipeline and returns the classifier and its schema.
    This function defines all the parameters for the pipeline steps, making them
    explicit and easy to change in one place.
    """
    logger.info("Starting full model preparation pipeline...")

    # Step 1: Load data
    logger.info(f"Step 1: Preparing to load data from '{file_path}'")
    data_handler = DataHandler(data_path=file_path)
    data_raw = data_handler.load_data()

    # Step 2: Clean data
    # The manager decides how to clean data for this pipeline.
    # Note: To handle missing values, we could add:
    # missing_value_strategy="fill", fill_value="Unknown"
    logger.info("Step 2: Cleaning data")
    data_cleaner = DataCleaner(columns_to_drop=["stalk-root"], remove_constants=True)
    data_cleaned = data_cleaner.clean(data_raw)

    # Step 3: Split data
    # The manager decides the splitting strategy.
    # To replicate original behavior, we want to validate the test set.
    logger.info(f"Step 3: Splitting data with '{target_col}' as target")
    data_splitter = DataSplitter(
        data=data_cleaned,
        target_col=target_col,
        test_size=0.3,
        random_state=42,
        validate_test_set=True,
    )
    train_df, test_df = data_splitter.split_data()

    # Step 4: Build the model artifact
    # The manager decides the model's hyperparameters.
    logger.info("Step 4: Building the Naive Bayes model artifact")
    model_builder = NaiveBayesModelBuilder(alpha=1.0)
    trained_model_artifact = model_builder.build_model(train_df, target_col=target_col)

    # Step 5: Evaluate performance using the self-contained evaluator
    # This step now happens before creating the final ClassifierService.
    logger.info("Step 5: Evaluating model performance using self-contained evaluator")
    accuracy_report = ModelEvaluatorService.run_evaluation(
        model_artifact=trained_model_artifact,
        test_data=test_df,
        target_col=target_col,
        pos_label=pos_label,
    )

    display_accuracy_report(accuracy_report, pos_label=pos_label)

    # Step 6: Check if accuracy meets the minimum threshold
    if accuracy_report["accuracy"] < min_accuracy:
        msg = (
            f"Model accuracy ({accuracy_report['accuracy']:.2%})"
            f" is below the minimum threshold of {min_accuracy:.2%}. Halting."
        )
        logger.error(msg)
        raise RuntimeError(msg)

    return trained_model_artifact