import pandas as pd
from typing import Dict, Any
from copy import deepcopy
from math import log


class NaiveBayesModelBuilder:
    """
    Builds a Naive Bayes model (log-probability weights) from training data.
    Its sole responsibility is to create the model artifact.
    """

    def __init__(self, alpha: float = 1.0):
        """
        Initializes the model builder with a smoothing parameter.

        Args:
            alpha (float): The smoothing parameter (Laplace).
        """
        self._alpha = alpha

    def build(self, train_data: pd.DataFrame, target_col: str) -> Dict[str, Any]:
        """
        Builds and returns the complete Naive Bayes model artifact (the weights).
        This method orchestrates the two-step process of counting and converting to weights.

        Args:
            train_data (pd.DataFrame): The training data.
            target_col (str): The name of the target column.

        Returns:
            Dict[str, Any]: The trained model artifact, containing log-probabilities.
        """
        # Step 1: Calculate raw counts from the data.
        counts_model = self._get_model_counts(train_data, target_col)

        # Step 2: Convert the raw counts into log-probability weights.
        weights_model = self._convert_counts_to_weights(counts_model)

        return weights_model

    # --- Private Helper Methods ---

    @staticmethod
    def _get_model_counts(data: pd.DataFrame, target_col: str) -> Dict[str, Any]:
        """
        Calculates frequency counts for all features, conditioned on the target variable.
        This is a static method as it does not depend on the instance's state.

        Returns:
            A dictionary containing likelihoods, target counts, and total count.
        """
        target_counts = {val: 0 for val in data[target_col].unique()}

        feature_cols = [col for col in data.columns if col != target_col]

        unique_value_dicts = {
            col: {val: 0 for val in data[col].unique()}
            for col in feature_cols
        }

        likelihoods = {
            val: deepcopy(unique_value_dicts)
            for val in data[target_col].unique()
        }

        for _, row in data.iterrows():
            target_value = row[target_col]
            target_counts[target_value] += 1
            for col in feature_cols:
                likelihoods[target_value][col][row[col]] += 1

        result = {
            "likelihoods": likelihoods,
            "target_counts": target_counts,
            "total_count": len(data),  # More direct than sum(target_counts.values())
        }

        return result

    def _convert_counts_to_weights(self, counts_model: dict) -> Dict[str, Any]:
        """
        Converts a model of raw counts into log-probabilities, applying smoothing.
        This method uses the instance's `_alpha` for smoothing.
        """
        likelihoods = counts_model["likelihoods"]
        target_counts = counts_model["target_counts"]
        total_count = counts_model["total_count"]

        weights = {}

        for target_value, column_dict in deepcopy(likelihoods).items():
            weights[target_value] = {}

            # Calculate the log prior probability for the current target class
            prior = log(target_counts[target_value] / total_count)
            weights[target_value]["__prior__"] = prior

            # Iterate through each feature for the current target class
            for column, value_dict in column_dict.items():
                # Conditionally apply smoothing if a zero count is detected
                if any(v == 0 for v in value_dict.values()):
                    value_dict = {k: v + self._alpha for k, v in value_dict.items()}

                total = sum(value_dict.values())

                # Calculate the log likelihood for each value
                weights[target_value][column] = {
                    k: log(v / total) for k, v in value_dict.items()
                }

        return weights