# app/classifier.py
from typing import Any, Dict, Hashable, Literal

from .logger_config import get_logger
from .model_artifact import IModelArtifact

logger = get_logger(__name__)

# הגדרת סוג חדש לאסטרטגיות האפשריות כדי לשפר קריאות ו-type-hinting
UnseenPredictStrategy = Literal["fail", "ignore_feature"]


class ClassifierService:
    """
    A service to perform predictions using a pre-trained model artifact.
    It relies on the IModelArtifact interface, and its behavior for unseen values
    can be configured during initialization.
    """

    def __init__(
        self,
        model_artifact: IModelArtifact,
        on_unseen_in_predict: UnseenPredictStrategy = "fail",  # <<< הוספת הפרמטר החדש >>>
    ):
        """
        Initializes the ClassifierService.

        Args:
            model_artifact (IModelArtifact): A trained model artifact that conforms
                                             to the IModelArtifact interface.
            on_unseen_in_predict (UnseenPredictStrategy): Strategy for handling
                unseen feature values during prediction.
                - "fail" (default): Raises a ValueError.
                - "ignore_feature": Ignores the feature with the unseen value and continues prediction.
        """
        if not model_artifact:
            msg = (
                "ClassifierService cannot be initialized with an empty model artifact."
            )
            logger.error(msg)
            raise ValueError(msg)

        self._model_artifact = model_artifact
        self._on_unseen_in_predict = on_unseen_in_predict  # שמירת האסטרטגיה
        logger.info(
            f"ClassifierService initialized with a model artifact. "
            f"Strategy for unseen values: '{self._on_unseen_in_predict}'"
        )

    def predict(self, sample: dict[Hashable, Any]) -> Dict[str, Any]:
        """
        Predicts the class for a single sample using the provided model artifact
        and the configured strategy for unseen values.
        """
        logger.debug(f"Starting prediction for sample: {sample}")

        best_class = None
        best_log_prob = float("-inf")
        class_scores = {}

        for target_value in self._model_artifact.get_all_class_labels():
            class_details = self._model_artifact.get_prediction_details(target_value)
            if not class_details:
                continue

            log_prob = class_details["__prior__"]

            for feature, value in sample.items():
                if feature not in class_details:
                    msg = (
                        f"Unseen feature encountered during prediction: '{feature}'. "
                        f"This feature was not in the training data."
                    )
                    logger.error(msg)
                    raise ValueError(msg)

                # <<< התניית הלוגיקה בפרמטר >>>
                if value not in class_details[feature]:
                    # אם האסטרטגיה היא להיכשל (ההתנהגות המקורית)
                    if self._on_unseen_in_predict == "fail":
                        msg = (
                            f"Unseen value for feature '{feature}': '{value}'. "
                            f"This value was not seen for this feature during training."
                        )
                        logger.error(msg)
                        raise ValueError(msg)

                    # אם האסטרטגיה היא להתעלם מהתכונה
                    elif self._on_unseen_in_predict == "ignore_feature":
                        logger.warning(
                            f"Ignoring unseen value '{value}' for feature '{feature}'"
                            f" during prediction, as per strategy."
                        )
                        # מדלגים על התכונה הזו וממשיכים לתכונה הבאה בלולאה
                        continue

                # הלוגיקה הזו תרוץ רק אם הערך קיים, או אם האסטרטגיה היא להתעלם
                log_prob += class_details[feature][value]

            class_scores[target_value] = log_prob
            if log_prob > best_log_prob:
                best_log_prob = log_prob
                best_class = target_value

        logger.debug(
            f"Prediction scores: {class_scores}. Final prediction: {best_class}"
        )
        return {"prediction": best_class}
