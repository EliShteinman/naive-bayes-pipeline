# backend/nb_classifier/classifier.py
from typing import Any, Dict, Hashable

from .logger_config import get_logger

logger = get_logger(__name__)


class ClassifierService:
    """
    A service to perform predictions using a pre-trained Naive Bayes model artifact.
    """

    def __init__(self, model_artifact: Dict[str, Any]):
        """
        Initializes the ClassifierService.

        Args:
            model_artifact (Dict[str, Any]): The trained model, which is a
                dictionary of log-probabilities.

        Raises:
            ValueError: If the model_artifact is missing or empty.
        """
        if not model_artifact:
            logger.error(
                "ClassifierService cannot be initialized with an empty model artifact."
            )
            raise ValueError(
                "ClassifierService must be initialized with a model artifact."
            )
        self._model = model_artifact
        logger.info("ClassifierService initialized successfully with a model artifact.")

    def predict(self, sample: dict[Hashable, Any]) -> Dict[str, Any]:
        """
        Predicts the class for a single sample using the Naive Bayes model.

        The prediction is made by calculating the log-probability for each class
        and choosing the class with the highest score.

        Args:
            sample (dict): A dictionary where keys are feature names and values
                           are the feature values for the sample.

        Returns:
            Dict[str, Any]: A dictionary containing the predicted class label.
                            Example: {"prediction": "poisonous"}

        Raises:
            ValueError: If the sample contains a feature or value that was not
                        seen during training.
        """
        logger.debug(f"Starting prediction for sample: {sample}")

        best_class = None
        best_log_prob = float("-inf")
        class_scores = {}

        # Iterate over each possible target class in the model
        for target_value in self._model.keys():
            # Start with the log prior probability of the class
            log_prob = self._model[target_value]["__prior__"]

            # Add the log likelihood for each feature in the sample
            for feature, value in sample.items():
                if feature not in self._model[target_value]:
                    msg = f"Unseen feature encountered during prediction: '{feature}'."\
                          f" This feature was not in the training data."
                    logger.error(msg)
                    raise ValueError(msg)

                if value not in self._model[target_value][feature]:
                    msg = f"Unseen value for feature '{feature}': '{value}'."\
                          f" This value was not seen for this feature during training."
                    logger.error(msg)
                    raise ValueError(msg)

                log_prob += self._model[target_value][feature][value]

            class_scores[target_value] = log_prob
            if log_prob > best_log_prob:
                best_log_prob = log_prob
                best_class = target_value

        logger.debug(
            f"Prediction scores: {class_scores}. Final prediction: {best_class}"
        )
        return {"prediction": best_class}
