import pandas as pd
from typing import Dict, Tuple, Union, List
from copy import deepcopy
from math import log
from config.logger_config import get_logger

logger = get_logger(__name__)


Target = str
Feature = str
Value = str
EmptyCountsTemplate = Dict[Feature, Dict[Value, int]]
Likelihoods = Dict[Target, EmptyCountsTemplate]
TargetCounts = Dict[Target, int]
Weights = Dict[Target, Dict[Feature, Dict[Value, float]]]
TargetPriors = Dict[Target, float]
FeatureValues = Dict[Feature, List[Value]]
Model = Dict[str, Union[FeatureValues, Weights, TargetPriors]]


class NaiveBayesModelBuilder:

    @staticmethod
    def build_model(data: pd.DataFrame, target_col: str) -> Model:
        df = data
        empty_counts_template = NaiveBayesModelBuilder._build_empty_counts_template(
            df, target_col
        )
        likelihoods, target_counts = (
            NaiveBayesModelBuilder._compute_feature_value_counts(
                df, target_col, empty_counts_template
            )
        )
        total_count = df.shape[0]
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
        return model

    @staticmethod
    def _build_empty_counts_template(
        df: pd.DataFrame, exclude_col: str
    ) -> EmptyCountsTemplate:
        features = NaiveBayesModelBuilder._get_feature_columns(df, exclude_col)
        empty_counts = {
            feature: {val: 0 for val in df[feature].unique()} for feature in features
        }
        return empty_counts

    @staticmethod
    def _compute_feature_value_counts(
        df: pd.DataFrame, target_col: str, template: EmptyCountsTemplate
    ) -> Tuple[Likelihoods, TargetCounts]:
        target_values = df[target_col].unique()
        likelihoods = {val: deepcopy(template) for val in target_values}
        grouped_by_target = df.groupby(target_col)
        feature_cols = NaiveBayesModelBuilder._get_feature_columns(df, target_col)
        for col in feature_cols:
            counts = grouped_by_target[col].value_counts().unstack(fill_value=0)
            counts_dict = counts.to_dict(orient="index")
            for target_val, value_counts in counts_dict.items():
                likelihoods[target_val][col].update(value_counts)
        target_counts = df[target_col].value_counts().to_dict()
        return likelihoods, target_counts

    @staticmethod
    def _calculate_likelihoods_and_priors(
        likelihoods: Likelihoods, target_counts: TargetCounts, total_count: int
    ) -> Tuple[Weights, TargetPriors]:
        weights = {}
        for target_val, feature_counts in deepcopy(likelihoods).items():
            weights[target_val] = {}
            for feature, value_counts in feature_counts.items():
                if any(v == 0 for v in value_counts.values()):
                    logger.warning(
                        f"Zero count found in feature '{feature}' for target '{target_val}'. This may affect model performance."
                    )
                    value_counts = {k: v + 1 for k, v in value_counts.items()}
                total_feature_count = sum(value_counts.values())
                weights[target_val][feature] = {
                    k: log(v / total_feature_count) for k, v in value_counts.items()
                }
        target_priors = {
            target_val: log(count / total_count)
            for target_val, count in target_counts.items()
        }
        return weights, target_priors

    @staticmethod
    def _get_feature_columns(df: pd.DataFrame, exclude_col: str) -> List[str]:
        return [col for col in df.columns if col != exclude_col]
