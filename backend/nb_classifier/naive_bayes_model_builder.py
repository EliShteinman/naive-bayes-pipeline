# backend/nb_classifier/naive_bayes_model_builder.py
from copy import deepcopy
from math import log
from typing import Any, Dict

import pandas as pd

from .logger_config import get_logger

logger = get_logger(__name__)


class NaiveBayesModelBuilder:
    """
    Builds a Naive Bayes model from training data.

    The model is represented as a dictionary of log-probabilities for
    efficient prediction.
    """

    def __init__(self, alpha: float = 1.0):
        """
        Initializes the model builder.

        Args:
            alpha (float): The smoothing parameter for Laplace/Lidstone smoothing.
                           A value of 1.0 corresponds to Laplace smoothing.
        """
        self._alpha = alpha
        logger.info(
            f"NaiveBayesModelBuilder initialized with alpha (smoothing) = {self._alpha}"
        )

    def build_model(
        self, train_data: pd.DataFrame, target_col: str, force_smoothing: bool = False
    ) -> Dict[str, Any]:
        """
        Orchestrates the model building process.

        Args:
            train_data (pd.DataFrame): The training data.
            target_col (str): The name of the target variable column.
            force_smoothing (bool): If True, applies smoothing to all features.
                                    If False, applies it only when a zero count is detected.

        Returns:
            Dict[str, Any]: The trained model as a dictionary of log-probabilities.
        """
        logger.info("Starting Naive Bayes model building process...")

        # Step 1: Calculate frequencies of all feature values and target classes.
        counts_model = self._get_model_counts(train_data, target_col)

        # Step 2: Convert these counts into smoothed log-probabilities.
        weights_model = self._convert_counts_to_weights(counts_model, force_smoothing)

        logger.info("Naive Bayes model built successfully.")
        return weights_model

    @staticmethod
    def _get_model_counts(data: pd.DataFrame, target_col: str) -> Dict[str, Any]:
        """
        Calculates the frequency counts for target classes and feature-value pairs.

        Args:
            data (pd.DataFrame): The training data.
            target_col (str): The name of the target variable column.

        Returns:
            Dict[str, Any]: A dictionary containing likelihoods, target counts,
                            and total sample count.
        """
        logger.info("Calculating frequency counts from training data...")
        feature_cols = [col for col in data.columns if col != target_col]

        # Initialize dictionaries to store counts
        target_values = data[target_col].unique()
        target_counts = {val: 0 for val in target_values}
        unique_value_dicts = {
            col: {val: 0 for val in data[col].unique()} for col in feature_cols
        }
        likelihoods = {val: deepcopy(unique_value_dicts) for val in target_values}

        # Iterate through data to populate counts
        for _, row in data.iterrows():
            target_value = row[target_col]
            target_counts[target_value] += 1
            for col in feature_cols:
                likelihoods[target_value][col][row[col]] += 1

        result = {
            "likelihoods": likelihoods,
            "target_counts": target_counts,
            "total_count": len(data),
        }
        logger.debug("Finished calculating model counts.")
        return result

    def _convert_counts_to_weights(
        self, counts_model: dict, force_smoothing: bool
    ) -> Dict[str, Any]:
        """
        Converts a model of raw counts into log-probabilities, applying smoothing.

        This method uses the instance's `_alpha` for smoothing. The output is a
        dictionary where values are log-probabilities, suitable for prediction via summation.

        Args:
            counts_model (dict): A dictionary of counts from `_get_model_counts`.
            force_smoothing (bool): If True, applies smoothing universally. Otherwise,
                                    applies it only to features with zero counts.

        Returns:
            Dict[str, Any]: The final model with log-probabilities.
        """
        logger.info("Converting counts to log-probabilities (weights)...")
        if force_smoothing:
            logger.info("Smoothing will be applied to all features.")
        else:
            logger.info("Smoothing will be applied only to features with zero counts.")

        likelihoods = counts_model["likelihoods"]
        target_counts = counts_model["target_counts"]
        total_count = counts_model["total_count"]

        weights = {}
        for target_value, column_dict in deepcopy(likelihoods).items():
            weights[target_value] = {}

            # Calculate the log prior probability for the current target class
            # P(class) = count(class) / total_count
            prior = log(target_counts[target_value] / total_count)
            weights[target_value]["__prior__"] = prior

            # Iterate through each feature for the current target class
            for column, value_dict in column_dict.items():

                # Conditionally apply smoothing based on the original logic
                # or if forced by the new parameter.
                should_smooth = force_smoothing or any(
                    v == 0 for v in value_dict.values()
                )

                if should_smooth:
                    # Apply Additive (Laplace/Lidstone) smoothing
                    value_dict_smoothed = {
                        k: v + self._alpha for k, v in value_dict.items()
                    }
                    total_smoothed = sum(value_dict_smoothed.values())

                    weights[target_value][column] = {
                        k: log(v / total_smoothed)
                        for k, v in value_dict_smoothed.items()
                    }
                else:
                    # Calculate standard log-likelihood without smoothing
                    total = sum(value_dict.values())
                    weights[target_value][column] = {
                        k: log(v / total) for k, v in value_dict.items() if v > 0
                    }

        logger.debug("Finished converting counts to weights.")
        return weights
