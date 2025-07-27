# app/naive_bayes_model_builder.py
from copy import deepcopy
from math import log
from typing import Any, Dict

import pandas as pd

from .logger_config import get_logger
from .model_artifact import IModelArtifact, NaiveBayesDictArtifact

logger = get_logger(__name__)


class NaiveBayesModelBuilder:
    """
    Builds a Naive Bayes model artifact from training data.
    This class encapsulates all the logic for counting frequencies,
    converting them to weights, and wrapping the result in a model artifact.
    """

    def __init__(self, alpha: float = 1.0):
        """
        Initializes the model builder.

        Args:
            alpha (float): The smoothing parameter for Laplace/Lidstone smoothing.
        """
        self._alpha = alpha
        logger.info(
            f"NaiveBayesModelBuilder initialized with alpha (smoothing) = {self._alpha}"
        )

    def build_model(
        self, train_data: pd.DataFrame, target_col: str, force_smoothing: bool = False
    ) -> IModelArtifact:
        """
        Orchestrates the model building process. This is the single public entry point.

        Args:
            train_data (pd.DataFrame): The training data.
            target_col (str): The name of the target variable column.
            force_smoothing (bool): Controls the smoothing strategy.

        Returns:
            IModelArtifact: The trained model artifact, ready for use.
        """
        logger.info("Starting Naive Bayes model building process...")

        # Step 1: Calculate frequencies of all feature values and target classes.
        counts_model = self._get_model_counts(train_data, target_col)

        # Step 2: Convert these counts into smoothed log-probabilities.
        weights_model_dict = self._convert_counts_to_weights(
            counts_model, force_smoothing
        )

        # Step 3: Wrap the final dictionary in a model artifact object.
        model_artifact = NaiveBayesDictArtifact(weights_model_dict)

        logger.info("Naive Bayes model built successfully.")
        return model_artifact

    # --- Internal (Private) Helper Methods ---

    def _get_model_counts(self, data: pd.DataFrame, target_col: str) -> Dict[str, Any]:
        """
        Calculates the frequency counts for target classes and feature-value pairs.
        This is a private helper method of the builder.
        """
        logger.info("Calculating frequency counts from training data...")
        feature_cols = [col for col in data.columns if col != target_col]

        target_values = data[target_col].unique()
        target_counts = {val: 0 for val in target_values}
        unique_value_dicts = {
            col: {val: 0 for val in data[col].unique()} for col in feature_cols
        }
        likelihoods = {val: deepcopy(unique_value_dicts) for val in target_values}

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
        This is a private helper method that uses the instance's alpha value.
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
            prior = log(target_counts[target_value] / total_count)
            weights[target_value]["__prior__"] = prior

            for column, value_dict in column_dict.items():
                should_smooth = force_smoothing or any(
                    v == 0 for v in value_dict.values()
                )
                if should_smooth:
                    value_dict_smoothed = {
                        k: v + self._alpha for k, v in value_dict.items()
                    }
                    total_smoothed = sum(value_dict_smoothed.values())
                    weights[target_value][column] = {
                        k: log(v / total_smoothed)
                        for k, v in value_dict_smoothed.items()
                    }
                else:
                    total = sum(value_dict.values())
                    weights[target_value][column] = {
                        k: log(v / total) for k, v in value_dict.items() if v > 0
                    }

        logger.debug("Finished converting counts to weights.")
        return weights
