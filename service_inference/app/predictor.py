# service_inference/app/model/predictor.py
from typing import Dict

import pandas as pd

from . import Model, Target, get_logger

logger = get_logger(__name__)


class Predictor:
    """
    A utility class to make predictions using a trained Naive Bayes model.
    """

    @staticmethod
    def _calculate_class_probability(
        instance: pd.Series, model: Model, target_class: Target
    ) -> float:
        """
        Calculates the log-probability of a single class for a given instance.

        Args:
            instance (pd.Series): A single row of data to predict.
            model (Model): The trained Naive Bayes model.
            target_class (Target): The specific class to calculate the probability for.

        Returns:
            float: The log-probability score for the given class.

        Raises:
            KeyError: If a feature or value in the instance is not found in the model's weights
                      for the given target class.
        """
        try:
            log_probability = model["target_priors"][target_class]
        except KeyError:
            raise ValueError(f"Target class '{target_class}' not found in model.")
        weights_for_all_classes = model["weights"]
        for feature_name, feature_value in instance.items():
            try:
                str_feature_name = str(feature_name)
                str_feature_value = str(feature_value)
                weight = weights_for_all_classes[target_class][str_feature_name][
                    str_feature_value
                ]
                log_probability += weight
            except KeyError:
                logger.error(
                    f"Unseen value '{feature_value}' for feature '{feature_name}' in class '{target_class}'."
                )
                raise KeyError(
                    f"Feature '{feature_name}' with value '{feature_value}'"
                    f" not found in model for class '{target_class}'."
                )

        return log_probability

    @staticmethod
    def predict_instance(instance: pd.Series, model: Model) -> Target:
        """
        Predicts the most likely class for a single data instance.

        Args:
            instance (pd.Series): A single row of data (features only) with encoded values.
            model (Model): The trained Naive Bayes model.

        Returns:
            Target: The predicted class name (encoded).

        Raises:
            KeyError: If any value in the instance is not recognized by the model.
                      This error is propagated from _calculate_class_probability.
        """
        possible_target_classes = model["target_priors"].keys()
        class_scores: Dict[str, float] = {}
        for current_class in possible_target_classes:
            score = Predictor._calculate_class_probability(
                instance, model, current_class
            )
            class_scores[current_class] = score
        if not class_scores:
            raise ValueError(
                "Could not calculate scores for any class. The model or input might be problematic."
            )

        best_class = max(class_scores, key=lambda k: class_scores[k])

        return best_class
