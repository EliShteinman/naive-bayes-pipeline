# service_inference/app/common/typing_defs.py
from typing import Dict, List, TypedDict

# --- Common Type Aliases ---
Target = str
Feature = str
Value = str

# --- Intermediate Model Structure Types ---
# These describe parts of the model during the build process
EmptyCountsTemplate = Dict[Feature, Dict[Value, int]]
Likelihoods = Dict[Target, EmptyCountsTemplate]
TargetCounts = Dict[Target, int]
Weights = Dict[Target, Dict[Feature, Dict[Value, float]]]
TargetPriors = Dict[Target, float]
FeatureValues = Dict[Feature, List[Value]]


# --- Final Model Structure using TypedDict ---
# This defines the exact "shape" of the final model dictionary.
# Now, any IDE or static checker can understand that a 'Model' object
# must have these three specific keys with these specific types.
class Model(TypedDict):
    """
    Represents the final, trained Naive Bayes model structure.

    Attributes:
        feature_values (FeatureValues): A mapping of each feature to its list of possible values.
        weights (Weights): The log-likelihoods P(value | class) for each feature.
        target_priors (TargetPriors): The log-priors P(class) for each target class.
    """

    feature_values: FeatureValues
    weights: Weights
    target_priors: TargetPriors
