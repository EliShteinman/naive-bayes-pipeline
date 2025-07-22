# backend/nb_classifier/model_evaluator.py
from typing import Any, Dict, Hashable, List
from .classifier import ClassifierService
from .logger_config import get_logger

logger = get_logger(__name__)


class ModelEvaluatorService:
    """
    Service for evaluating the performance of a ClassifierService instance.
    """

    def __init__(self, classifier: ClassifierService):
        """
        Initializes the ModelEvaluatorService.

        Args:
            classifier (ClassifierService): An initialized classifier instance
                                            that will be evaluated.

        Raises:
            ValueError: If the classifier is not provided.
        """
        if not classifier:
            logger.error(
                "ModelEvaluatorService must be initialized with a ClassifierService instance."
            )
            raise ValueError(
                "ModelEvaluatorService must be initialized with a ClassifierService instance."
            )
        self._classifier = classifier
        logger.info("ModelEvaluatorService initialized.")

    def run_evaluation(
        self, test_data: List[dict[Hashable, Any]], target_col: str
    ) -> Dict[str, Any]:
        """
        Runs the evaluation process on a test dataset.

        Iterates through the test data, makes a prediction for each sample,
        compares it to the true label, and calculates the overall accuracy.

        Args:
            test_data (List[dict[Hashable, Any]]): A list of dictionaries, where
                                                   each dictionary is a test sample.
            target_col (str): The name of the column containing the true labels.

        Returns:
            Dict[str, Any]: A report dictionary containing accuracy, total samples,
                            correct predictions, and incorrect predictions.
        """
        logger.info(f"Starting model evaluation on {len(test_data)} samples.")

        if not test_data:
            logger.warning(
                "Evaluation requested on an empty test dataset. Returning zero accuracy."
            )
            return {
                "accuracy": 0,
                "total_samples": 0,
                "correct_predictions": 0,
                "incorrect_predictions": 0,
            }

        correct = 0
        total = len(test_data)

        for i, row in enumerate(test_data):
            true_label = row[target_col]
            sample = {k: v for k, v in row.items() if k != target_col}

            try:
                pred_label = self._classifier.predict(sample)["prediction"]
                if pred_label == true_label:
                    correct += 1
                logger.debug(
                    f"Sample {i + 1}/{total}: True='{true_label}', Predicted='{pred_label}'. Correct: {pred_label == true_label}"
                )
            except ValueError as e:
                logger.error(
                    f"Failed to predict sample {i + 1} due to an error: {e}. This sample will be skipped in accuracy calculation."
                )
                # We decrement total because this sample could not be evaluated.
                total -= 1

        if total == 0:
            logger.warning("No samples could be evaluated. Returning zero accuracy.")
            return {
                "accuracy": 0,
                "total_samples": len(test_data),
                "correct_predictions": 0,
                "incorrect_predictions": len(test_data),
            }

        accuracy = correct / total
        report = {
            "accuracy": accuracy,
            "total_samples": total,
            "correct_predictions": correct,
            "incorrect_predictions": total - correct,
        }

        logger.info(f"Evaluation finished. Accuracy: {accuracy:.2%}")
        return report
