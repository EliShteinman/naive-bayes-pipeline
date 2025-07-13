# nb_classifier/naive_bayes_model_builder.py
import pandas as pd
from typing import Dict, Any, List
from copy import deepcopy
from math import log
from nb_classifier.logger_config import get_logger

logger = get_logger(__name__)
class NaiveBayesModelBuilder:
    def __init__(self, alpha: float = 1.0):
        self._alpha = alpha

    def build_model(self, train_data: pd.DataFrame, target_col: str) -> Dict[str, Any]:
        df = train_data
        counts_model = self._get_model_counts(df, target_col)
        weights_model = self._convert_counts_to_weights(counts_model)
        return weights_model


    @staticmethod
    def _get_model_counts(data: pd.DataFrame, target_col: str) -> Dict[str, Any]:
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
        logger.debug(f"Model counts: {result}")
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
        logger.debug(f"Converted counts to weights: {weights}")
        return weights