# nb_classifier/model_evaluator.py
from typing import Any, Dict, Hashable, List

from .classifier import ClassifierService
from .logger_config import get_logger

logger = get_logger(__name__)


class ModelEvaluatorService:

    def __init__(self, classifier: ClassifierService):

        if not classifier:
            logger.error("ClassifierService must be initialized with a model artifact.")
            raise ValueError(
                "ClassifierService must be initialized with a model artifact."
            )
        self._classifier: ClassifierService = classifier

    def run_evaluation(
        self, test_data: List[dict[Hashable, Any]], target_col: str
    ) -> Dict[str, Any]:
        correct = 0
        total = len(test_data)
        for row in test_data:
            true_label = row[target_col]
            sample = {k: v for k, v in row.items() if k != target_col}

            pred_label = self._classifier.predict(sample)["prediction"]
            if pred_label == true_label:
                correct += 1

        accuracy = correct / total if total > 0 else 0
        return {
            "accuracy": accuracy,
            "total_samples": total,
            "correct_predictions": correct,
            "incorrect_predictions": total - correct,
        }
