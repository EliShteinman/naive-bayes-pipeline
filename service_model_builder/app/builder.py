# service_training/app/model/builder.py
from copy import deepcopy
from math import log
from typing import List, Optional

import pandas as pd

from .common import (
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
    An expert builder object that constructs a Naive Bayes model.
    It holds the state of the build process internally.
    """

    def __init__(self, alpha: int = 1):
        """
        Initializes the model builder.

        Args:
            alpha (int): The smoothing factor for Laplace smoothing. Default is 1.
        """
        logger.info(f"NaiveBayesModelBuilder instance created with alpha={alpha}.")
        self._alpha = alpha
        self._data: Optional[pd.DataFrame] = None
        self._target_col: Optional[Target] = None
        self._feature_cols: Optional[List[Feature]] = None
        self._empty_counts_template: Optional[EmptyCountsTemplate] = None
        self._likelihoods: Optional[Likelihoods] = None
        self._target_counts: Optional[TargetCounts] = None
        self._total_count: Optional[int] = None
        self._weights: Optional[Weights] = None
        self._target_priors: Optional[TargetPriors] = None

    def build(self, data: pd.DataFrame, target_col: Target) -> Model:
        """
        Runs the full model building process.

        This is the main method to call. It runs all the steps in the correct order.

        Args:
            data (pd.DataFrame): The training data.
            target_col (Target): The name of the column to predict.

        Returns:
            Model: The final, trained model dictionary.
        """
        logger.info("--- Starting model build process ---")
        self._initialize_build(data, target_col)
        self._build_empty_counts_template()
        self._compute_feature_value_counts()
        self._calculate_likelihoods_and_priors()
        final_model = self._assemble_model()

        logger.info("--- Model build process finished successfully. ---")
        return final_model

    def _initialize_build(self, data: pd.DataFrame, target_col: Target) -> None:
        """
        Sets up the initial state for the build process.

        It saves the data and config inside the object.
        """
        logger.info(
            f"Step 1: Initializing build for target '{target_col}'. Data shape: {data.shape}"
        )
        if target_col not in data.columns:
            logger.error(f"Target column '{target_col}' not found in data. Aborting.")
            raise ValueError(f"Target column '{target_col}' not found.")

        self._data = data
        self._target_col = target_col
        self._feature_cols = [
            col for col in self._data.columns if col != self._target_col
        ]
        self._total_count = self._data.shape[0]
        logger.debug(f"Found {len(self._feature_cols)} feature columns.")

    def _build_empty_counts_template(self) -> None:
        """
        Creates an empty structure to store counts.

        The structure looks like: { "feature1": {"valueA": 0, "valueB": 0}, ... }
        """
        logger.info("Step 2: Building empty counts template.")
        assert (
            self._data is not None and self._feature_cols is not None
        ), "Initialization must run first."

        self._empty_counts_template = {
            feature: {str(val): 0 for val in self._data[feature].unique()}
            for feature in self._feature_cols
        }
        logger.debug("Empty template created successfully.")

    def _compute_feature_value_counts(self) -> None:
        """
        Counts how many times each feature value appears for each target class.

        For example, it counts how many times `color=red` appears when `fruit=apple`.
        """
        logger.info("Step 3: Computing feature value counts.")
        assert (
            self._data is not None and self._target_col is not None
        ), "Initialization must run first."
        assert self._empty_counts_template is not None, "Template must be built first."

        target_values = self._data[self._target_col].unique()
        logger.debug(f"Found target classes: {list(target_values)}")
        self._likelihoods = {
            val: deepcopy(self._empty_counts_template) for val in target_values
        }

        grouped_by_target = self._data.groupby(self._target_col)

        for col in self._feature_cols:
            # This line is complex. It groups by target and counts values in the column.
            counts = grouped_by_target.apply(lambda x: x[col].value_counts()).unstack(
                fill_value=0
            )
            counts_dict = counts.to_dict(orient="index")

            for target_val, value_counts in counts_dict.items():
                if target_val in self._likelihoods:
                    # str() to make sure all keys are strings
                    self._likelihoods[target_val][col].update(
                        {str(k): v for k, v in value_counts.items()}
                    )

        self._target_counts = self._data[self._target_col].value_counts().to_dict()
        logger.debug("Finished computing all counts.")

    def _calculate_likelihoods_and_priors(self) -> None:
        """
        Calculates the final probabilities (weights) for the model.

        It converts the raw counts into log probabilities using the Naive Bayes formula.
        """
        logger.info("Step 4: Calculating log-likelihoods and target priors.")
        assert (
            self._likelihoods is not None and self._target_counts is not None
        ), "Counts must be computed first."

        weights = {}
        temp_likelihoods = deepcopy(self._likelihoods)

        for target_val, feature_counts in temp_likelihoods.items():
            weights[target_val] = {}
            for feature, value_counts in feature_counts.items():
                # Apply Laplace smoothing if any count is zero
                if any(v == 0 for v in value_counts.values()):
                    logger.warning(
                        f"Found zero count in feature '{feature}' for target '{target_val}'. Applying smoothing."
                    )
                    value_counts = {k: v + self._alpha for k, v in value_counts.items()}

                total_feature_count = sum(value_counts.values())
                weights[target_val][feature] = {}
                for k, v in value_counts.items():
                    try:
                        weights[target_val][feature][k] = log(v / total_feature_count)
                    except ValueError:
                        logger.error(
                            f"Math error for feature '{feature}', value '{k}'. Check for zero counts."
                        )
                        weights[target_val][feature][
                            k
                        ] = -999.0  # Assign a large negative number

        self._weights = weights
        logger.debug("Calculated all feature weights.")

        self._target_priors = {
            target_val: log(count / self._total_count)
            for target_val, count in self._target_counts.items()
        }
        logger.debug("Calculated all target priors.")

    def _assemble_model(self) -> Model:
        """
        Puts all the calculated parts together into the final model dictionary.

        Returns:
            Model: The final, complete model dictionary.
        """
        logger.info("Step 5: Assembling the final model.")
        # Assertions to make sure all parts are ready before assembly.
        assert self._weights is not None, "Weights were not calculated."
        assert self._target_priors is not None, "Target priors were not calculated."
        assert self._empty_counts_template is not None, "Template was not created."

        feature_values = {
            feature: list(values_dict.keys())
            for feature, values_dict in self._empty_counts_template.items()
        }

        model: Model = {
            "feature_values": feature_values,
            "weights": self._weights,
            "target_priors": self._target_priors,
        }
        logger.debug("Final model dictionary assembled.")
        return model
