# model/naive_bayes_builder.py
from copy import deepcopy
from math import log
from typing import List, Tuple

import pandas as pd

from nb_classifier import (
    EmptyCountsTemplate,
    Feature,
    Likelihoods,
    Model,
    Target,
    TargetCounts,
    TargetPriors,
    Weights,
    get_logger,
)

logger = get_logger(__name__)


class NaiveBayesModelBuilder:
    """
    A utility class with static methods to build a categorical Naive Bayes model.

    This class does not need to be instantiated. All methods are static and can be
    called directly, e.g., `NaiveBayesModelBuilder.build_model(...)`.
    """

    @staticmethod
    def build_model(data: pd.DataFrame, target_col: Target) -> Model:
        """
        Builds a full Naive Bayes model from the provided data.

        This is the main public method that orchestrates the entire model-building
        process. It calls helper methods to calculate counts, probabilities,
        and assemble the final model dictionary.

        Args:
            data (pd.DataFrame): The training data. It should contain categorical
                                 features and the target column.
            target_col (str): The name of the column to be predicted.

        Returns:
            Model (Dict): A dictionary representing the trained model. It contains:
                          - "feature_values": A map of features to their possible values.
                          - "weights": The calculated log-likelihoods for each feature-value pair.
                          - "target_priors": The calculated log-prior for each target class.

        Raises:
            ValueError: If the `target_col` is not found in the DataFrame's columns.
        """
        logger.info(
            f"Starting Naive Bayes model build for target column: '{target_col}'"
        )
        logger.info(f"Input data shape: {data.shape[0]} rows, {data.shape[1]} columns.")

        if target_col not in data.columns:
            logger.error(
                f"Target column '{target_col}' not found in DataFrame columns. Aborting build."
            )
            raise ValueError(f"Target column '{target_col}' not found.")

        df = data

        logger.info("Step 1: Building empty counts template for features.")
        empty_counts_template = NaiveBayesModelBuilder._build_empty_counts_template(
            df, target_col
        )

        logger.info(
            "Step 2: Computing feature value counts based on the target variable."
        )
        likelihoods, target_counts = (
            NaiveBayesModelBuilder._compute_feature_value_counts(
                df, target_col, empty_counts_template
            )
        )

        total_count = df.shape[0]
        logger.info("Step 3: Calculating log-likelihoods and target priors.")
        weights, target_priors = (
            NaiveBayesModelBuilder._calculate_likelihoods_and_priors(
                likelihoods, target_counts, total_count
            )
        )

        feature_values = {
            feature: list(value_dict.keys())
            for feature, value_dict in empty_counts_template.items()
        }

        model = {
            "feature_values": feature_values,
            "weights": weights,
            "target_priors": target_priors,
        }
        logger.info("Naive Bayes model built successfully.")
        return model

    @staticmethod
    def _build_empty_counts_template(
        df: pd.DataFrame, exclude_col: Target
    ) -> EmptyCountsTemplate:
        """
        Creates a nested dictionary structure to hold feature value counts.

        This helper method generates a template like:
        {
            "feature1": {"valueA": 0, "valueB": 0},
            "feature2": {"valueX": 0, "valueY": 0}
        }
        This structure is then used to store counts for each target class.

        Args:
            df (pd.DataFrame): The input DataFrame.
            exclude_col (str): The name of the column to exclude (usually the target column).

        Returns:
            EmptyCountsTemplate (Dict): A nested dictionary with all counts initialized to zero.
        """
        features = NaiveBayesModelBuilder._get_feature_columns(df, exclude_col)
        logger.debug(f"Building template for {len(features)} features.")
        empty_counts = {
            feature: {val: 0 for val in df[feature].unique()} for feature in features
        }
        return empty_counts

    @staticmethod
    def _compute_feature_value_counts(
        df: pd.DataFrame, target_col: Target, template: EmptyCountsTemplate
    ) -> Tuple[Likelihoods, TargetCounts]:
        """
        Calculates the raw counts of each feature value for each target class.

        It groups the data by the target column and then counts the occurrences
        of each value within each feature.

        Args:
            df (pd.DataFrame): The input DataFrame.
            target_col (str): The name of the target column.
            template (EmptyCountsTemplate): The empty structure to be filled with counts.

        Returns:
            Tuple[Likelihoods, TargetCounts]: A tuple containing two items:
                - Likelihoods (Dict): A dictionary with raw counts, e.g.,
                  `{"targetA": {"feature1": {"valueX": 10, ...}}}`.
                - TargetCounts (Dict): A dictionary with the total count of each target class,
                  e.g., `{"targetA": 100, "targetB": 150}`.
        """
        target_values = df[target_col].unique()
        logger.debug(f"Identified target values: {list(target_values)}")

        likelihoods = {val: deepcopy(template) for val in target_values}
        grouped_by_target = df.groupby(target_col)
        feature_cols = NaiveBayesModelBuilder._get_feature_columns(df, target_col)

        logger.info("Calculating value counts for each feature, grouped by target...")
        for col in feature_cols:
            counts = grouped_by_target.apply(lambda x: x[col].value_counts()).unstack(
                fill_value=0
            )
            counts_dict = counts.to_dict(orient="index")
            for target_val, value_counts in counts_dict.items():
                if target_val in likelihoods:
                    likelihoods[target_val][col].update(value_counts)

        target_counts = df[target_col].value_counts().to_dict()
        logger.info("Finished computing feature value counts.")
        return likelihoods, target_counts

    @staticmethod
    def _calculate_likelihoods_and_priors(
        likelihoods: Likelihoods, target_counts: TargetCounts, total_count: int
    ) -> Tuple[Weights, TargetPriors]:
        """
        Converts raw counts into log-probabilities (weights) and log-priors.

        This method applies the Naive Bayes formulas to calculate:
        1. P(Feature=Value | Target=Class) -> Log-Likelihoods
        2. P(Target=Class) -> Log-Priors

        It also applies Laplace (add-1) smoothing if a zero count is found
        to avoid taking the log of zero.

        Args:
            likelihoods (Likelihoods): The raw counts of feature values per target class.
            target_counts (TargetCounts): The total count for each target class.
            total_count (int): The total number of rows in the dataset.

        Returns:
            Tuple[Weights, TargetPriors]: A tuple containing:
                - Weights (Dict): A dictionary of log-likelihoods.
                - TargetPriors (Dict): A dictionary of log-priors for each target class.
        """
        weights = {}
        temp_likelihoods = deepcopy(likelihoods)

        for target_val, feature_counts in temp_likelihoods.items():
            weights[target_val] = {}
            for feature, value_counts in feature_counts.items():
                if any(v == 0 for v in value_counts.values()):
                    logger.warning(
                        f"Zero count found in feature '{feature}' for target '{target_val}'. "
                        f"Applying Laplace (add-1) smoothing to prevent log(0)."
                    )
                    value_counts = {k: v + 1 for k, v in value_counts.items()}

                total_feature_count = sum(value_counts.values())

                weights[target_val][feature] = {}
                for k, v in value_counts.items():
                    try:
                        weights[target_val][feature][k] = log(v / total_feature_count)
                    except ValueError as e:
                        logger.error(
                            f"Math domain error for feature='{feature}', value='{k}', target='{target_val}'. "
                            f"Count: {v}, Total Count: {total_feature_count}. Error: {e}. "
                            "Setting weight to a large negative number."
                        )
                        weights[target_val][feature][k] = -999.0

        logger.info("Calculating target priors.")
        target_priors = {
            target_val: log(count / total_count)
            for target_val, count in target_counts.items()
        }
        return weights, target_priors

    @staticmethod
    def _get_feature_columns(df: pd.DataFrame, exclude_col: Target) -> List[Feature]:
        """
        Gets a list of all column names from a DataFrame, except for one.

        Args:
            df (pd.DataFrame): The input DataFrame.
            exclude_col (str): The name of the column to exclude from the list.

        Returns:
            List[str]: A list of column names that are considered features.
        """
        features = [col for col in df.columns if col != exclude_col]
        logger.debug(f"Identified {len(features)} feature columns: {features}")
        return features
