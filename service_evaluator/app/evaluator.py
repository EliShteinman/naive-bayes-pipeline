# service_evaluator/app/evaluator.py

from typing import Any, Dict, List

import pandas as pd

from .common.logger_config import get_logger
from .common.typing_defs import Model, Target

logger = get_logger(__name__)


class ModelEvaluator:
    """
    A utility class to evaluate a model's performance on a test dataset.
    This class is stateless and all its methods perform calculations from scratch.
    """

    def __init__(self):
        """Initializes the ModelEvaluator."""
        logger.debug("ModelEvaluator instance created.")
        pass

    def evaluate(
        self, model: Model, test_data: pd.DataFrame, target_col: Target
    ) -> Dict[str, float]:
        x_test = test_data.drop(columns=[target_col])
        y_target = test_data[target_col]
        y_pred = x_test.apply(lambda row: self._predict_instance(model, row), axis=1)
        metrics = self._calculate_metrics(y_target, y_pred, model)
        return metrics


    def _predict_batch(self, model: Model, data: pd.DataFrame) -> List[Target]:
        """
        Predicts the class for each row in the input DataFrame.
        This method must be implemented from scratch.

        Args:
            model (Model): The trained model.
            data (pd.DataFrame): A DataFrame containing only the feature columns.

        Returns:
            List[Target]: A list of predicted class labels for each row.
        """
        return [self._predict_instance(model, row) for _, row in data.iterrows()]

    def _predict_instance(self, model: Model, instance: pd.Series) -> Target:
        class_scores = {}
        for target_class in model["target_priors"]:
            score = self._calculate_class_probability(model, instance, target_class)
            class_scores[target_class] = score
        best_class = max(class_scores, key=lambda k: class_scores[k])
        return best_class

    def _calculate_class_probability(
        self, model: Model, instance: pd.Series, target_class: Target
    ) -> float:
        pass

    def _calculate_metrics(
        self, y_true: pd.Series, y_pred: List[Target], model: Model
    ) -> Dict[str, float]:
        """
        Calculates a standard set of classification metrics from scratch.

        This method will call helper methods for each specific metric.

        Args:
            y_true (pd.Series): The actual, true labels.
            y_pred (List[Target]): The labels predicted by the model.
            model (Model): The trained model, used to get the list of all possible labels.

        Returns:
            Dict[str, float]: A dictionary of performance metrics.
        """
        accuracy = self._calculate_accuracy(y_true, y_pred)
        labels = list(model["target_priors"].keys())
        confusion_matrix = self._build_confusion_matrix(y_true, y_pred, labels)
        pr = self._calculate_precision_and_recall(confusion_matrix)
        precision = pr["precision_weighted"]
        recall = pr["recall_weighted"]
        f1 = self._calculate_f1_score(precision, recall)
        return {
            "accuracy": accuracy,
            "precision_weighted": precision,
            "recall_weighted": recall,
            "f1_score": f1
        }

    def _calculate_accuracy(self, y_true: pd.Series, y_pred: List[Target]) -> float:
        """
        Calculates the accuracy score from scratch.
        Accuracy = (Number of Correct Predictions) / (Total Number of Predictions)

        Args:
            y_true (pd.Series): The true labels.
            y_pred (List[Target]): The predicted labels.

        Returns:
            float: The accuracy score, between 0.0 and 1.0.
        """
        pass

    def _build_confusion_matrix(
        self, y_true: pd.Series, y_pred: List[Target], labels: List[Target]
    ) -> Dict[Target, Dict[Target, int]]:
        """
        Builds a confusion matrix from scratch as a nested dictionary.
        The matrix structure will be: { 'true_label_A': {'pred_label_A': count, 'pred_label_B': count}, ... }

        Args:
            y_true (pd.Series): The true labels.
            y_pred (List[Target]): The predicted labels.
            labels (List[Target]): A list of all possible class labels.

        Returns:
            Dict[Target, Dict[Target, int]]: The confusion matrix.
        """
        pass

    def _calculate_precision_and_recall(
        self, confusion_matrix: Dict[Target, Dict[Target, int]]
    ) -> Dict[str, float]:
        """
        Calculates weighted precision and recall from a confusion matrix.
        This must be implemented from scratch.

        For each class, calculate:
        - Precision = TP / (TP + FP)
        - Recall = TP / (TP + FN)
        Then, calculate the weighted average based on the number of true instances for each class.

        Args:
            confusion_matrix (Dict): The confusion matrix built by _build_confusion_matrix.

        Returns:
            Dict[str, float]: A dictionary containing 'precision_weighted' and 'recall_weighted'.
        """
        pass

    def _calculate_f1_score(self, precision: float, recall: float) -> float:
        """
        Calculates the F1 score from precision and recall.
        F1 = 2 * (Precision * Recall) / (Precision + Recall)

        Args:
            precision (float): The calculated weighted precision.
            recall (float): The calculated weighted recall.

        Returns:
            float: The weighted F1 score.
        """
        pass
