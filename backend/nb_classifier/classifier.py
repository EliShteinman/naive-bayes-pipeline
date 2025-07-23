# backend/nb_classifier/classifier.py
from typing import Any, Dict, Hashable

from .logger_config import get_logger
from .model_artifact import IModelArtifact

logger = get_logger(__name__)


class ClassifierService:
    """
    A service to perform predictions using a pre-trained model artifact.
    It relies on the IModelArtifact interface, making it independent of the
    underlying model's implementation.
    """

    def __init__(self, model_artifact: IModelArtifact):
        """
        Initializes the ClassifierService.

        Args:
            model_artifact (IModelArtifact): A trained model artifact that conforms
                                             to the IModelArtifact interface.

        Raises:
            ValueError: If the model_artifact is missing.
        """
        if not model_artifact:
            msg = "ClassifierService cannot be initialized with an empty model artifact."
            logger.error(msg)
            raise ValueError(msg)
        self._model_artifact = model_artifact
        logger.info("ClassifierService initialized successfully with a model artifact.")

    def predict(self, sample: dict[Hashable, Any]) -> Dict[str, Any]:
        """
        Predicts the class for a single sample using the provided model artifact.

        The prediction is made by calculating the log-probability for each class
        and choosing the class with the highest score.

        Args:
            sample (dict): A dictionary representing the sample to predict.

        Returns:
            Dict[str, Any]: A dictionary containing the predicted class label.

        Raises:
            ValueError: If the sample contains a feature or value not in the model.
        """
        logger.debug(f"Starting prediction for sample: {sample}")

        best_class = None
        best_log_prob = float("-inf")
        class_scores = {}

        # CHANGED: Use the artifact's methods instead of direct dict access
        for target_value in self._model_artifact.get_all_class_labels():
            # Get the specific details (probabilities) for the current class
            class_details = self._model_artifact.get_prediction_details(target_value)
            if not class_details:
                continue # Should not happen with a valid artifact, but good practice

            # Start with the log prior probability of the class
            log_prob = class_details["__prior__"]

            # Add the log likelihood for each feature in the sample
            for feature, value in sample.items():
                if feature not in class_details:
                    msg = (f"Unseen feature encountered during prediction: '{feature}'. "
                           f"This feature was not in the training data.")
                    logger.error(msg)
                    raise ValueError(msg)

                if value not in class_details[feature]:
                    msg = (f"Unseen value for feature '{feature}': '{value}'. "
                           f"This value was not seen for this feature during training.")
                    logger.error(msg)
                    raise ValueError(msg)

                log_prob += class_details[feature][value]

            class_scores[target_value] = log_prob
            if log_prob > best_log_prob:
                best_log_prob = log_prob
                best_class = target_value

        logger.debug(
            f"Prediction scores: {class_scores}. Final prediction: {best_class}"
        )
        return {"prediction": best_class}