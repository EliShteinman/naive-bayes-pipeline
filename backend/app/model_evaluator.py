# app/model_evaluator.py
from typing import Any, Dict

import pandas as pd

from backend.app.logger_config import get_logger
from backend.app.model_artifact import IModelArtifact

logger = get_logger(__name__)


class ModelEvaluatorService:
    """
    A self-contained service for evaluating the performance of a model artifact
    on a test dataset. It uses efficient pandas operations for prediction.
    """

    @staticmethod
    def _predict_single_row(row: pd.Series, model_artifact: IModelArtifact) -> Any:
        """
        A helper function to predict a single row (pd.Series).
        Designed to be used with df.apply().
        """
        sample = row.to_dict()
        best_class = None
        best_log_prob = float("-inf")

        try:
            for target_value in model_artifact.get_all_class_labels():
                class_details = model_artifact.get_prediction_details(target_value)
                if not class_details:
                    continue

                log_prob = class_details["__prior__"]

                for feature, value in sample.items():
                    if feature not in class_details:
                        raise ValueError(f"Feature '{feature}' not in model artifact.")
                    if value not in class_details[feature]:
                        raise ValueError(
                            f"Value '{value}' for feature '{feature}' not in model artifact."
                        )

                    log_prob += class_details[feature][value]

            if log_prob > best_log_prob:
                best_log_prob = log_prob
                best_class = target_value

            return best_class

        except ValueError as e:
            logger.error(f"Could not predict for sample: {sample}. Error: {e}")
            return None

    @staticmethod
    def run_evaluation(
        model_artifact: IModelArtifact,
        test_data: pd.DataFrame,
        target_col: str,
        pos_label: Any,
    ) -> Dict[str, Any]:
        """
        Runs the full evaluation process on a test DataFrame using a model artifact.
        This method uses df.apply() for efficient batch prediction.
        """
        logger.info(
            f"Starting self-contained, efficient model evaluation on {len(test_data)} samples."
        )

        if test_data.empty:
            logger.warning("Evaluation requested on an empty test dataset.")
            return {
                "accuracy": 0,
                "precision": 0,
                "recall": 0,
                "f1_score": 0,
                "total_samples": 0,
                "correct_predictions": 0,
                "incorrect_predictions": 0,
            }

        true_labels = test_data[target_col]
        features_df = test_data.drop(columns=[target_col])

        predicted_labels = features_df.apply(
            ModelEvaluatorService._predict_single_row,
            axis=1,
            model_artifact=model_artifact,
        )

        valid_indices = predicted_labels.notna()
        true_labels_filtered = true_labels[valid_indices]
        predicted_labels_filtered = predicted_labels[valid_indices]

        if len(true_labels_filtered) == 0:
            logger.error("No samples could be predicted. Cannot evaluate.")
            return {
                "accuracy": 0,
                "precision": 0,
                "recall": 0,
                "f1_score": 0,
                "total_samples": len(test_data),
            }

        total_samples_evaluated = len(true_labels_filtered)

        tp = (
            (predicted_labels_filtered == pos_label)
            & (true_labels_filtered == pos_label)
        ).sum()
        fp = (
            (predicted_labels_filtered == pos_label)
            & (true_labels_filtered != pos_label)
        ).sum()
        tn = (
            (predicted_labels_filtered != pos_label)
            & (true_labels_filtered != pos_label)
        ).sum()
        fn = (
            (predicted_labels_filtered != pos_label)
            & (true_labels_filtered == pos_label)
        ).sum()

        correct_predictions = tp + tn
        accuracy = (
            correct_predictions / total_samples_evaluated
            if total_samples_evaluated > 0
            else 0
        )
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        report = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "total_samples": total_samples_evaluated,
            "correct_predictions": int(correct_predictions),
            "incorrect_predictions": int(total_samples_evaluated - correct_predictions),
        }

        logger.info(
            f"Self-contained evaluation finished. Accuracy: {accuracy:.2%}, F1-Score: {f1_score:.2f}"
        )
        return report
