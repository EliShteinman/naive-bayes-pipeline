# nb_classifier/classifier.py
from typing import Any, Dict, Hashable

from .logger_config import get_logger

logger = get_logger(__name__)


class ClassifierService:

    def __init__(self, model_artifact: Dict[str, Any]):

        if not model_artifact:
            logger.error("ClassifierService must be initialized with a model artifact.")
            raise ValueError(
                "ClassifierService must be initialized with a model artifact."
            )
        self._model = model_artifact

    def predict(self, sample: dict[Hashable, Any]) -> Dict[str, Any]:

        best_class = None
        best_log_prob = float("-inf")

        for target_value in self._model.keys():
            log_prob = self._model[target_value]["__prior__"]

            for feature, value in sample.items():
                if value not in self._model[target_value].get(feature, {}):
                    logger.error(
                        f"Unseen value encountered: {feature} = {value} for target class {target_value}"
                    )
                    raise ValueError(f"Unseen value encountered: {feature} = {value}")
                log_prob += self._model[target_value][feature][value]

            if log_prob > best_log_prob:
                best_log_prob = log_prob
                best_class = target_value
        return {"prediction": best_class}
