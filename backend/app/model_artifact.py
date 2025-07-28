# app/model_artifact.py
from abc import ABC, abstractmethod
from typing import Any, Dict, List


class IModelArtifact(ABC):
    """
    Abstract base class (Interface) for a trained model artifact.
    It defines the contract for what a model must be able to do,
    regardless of its internal implementation (e.g., dict, sklearn object).
    """

    @abstractmethod
    def get_prediction_details(self, class_label: Any) -> Dict:
        """Returns the internal details needed for prediction for a specific class."""
        pass

    @abstractmethod
    def get_all_class_labels(self) -> List:
        """Returns a list of all possible class labels."""
        pass

    @abstractmethod
    def get_schema(self) -> Dict[str, List[str]]:
        """Extracts and returns the feature schema from the model."""
        pass

    @abstractmethod
    def to_dict(self) -> Dict:
        """
        Returns a serializable dictionary representation of the model's data.
        """
        pass


class NaiveBayesDictArtifact(IModelArtifact):
    """A concrete implementation of IModelArtifact that wraps a dictionary-based Naive Bayes model."""

    def __init__(self, model_dict: Dict):
        if not model_dict:
            raise ValueError("Model dictionary cannot be empty.")
        self._model = model_dict

    def get_prediction_details(self, class_label: Any) -> Dict:
        return self._model.get(class_label)

    def get_all_class_labels(self) -> List:
        return list(self._model.keys())

    def get_schema(self) -> Dict[str, List[str]]:
        # This is the logic from the old 'extract_expected_features'
        first_class_dict = next(iter(self._model.values()))
        return {
            feature: sorted(value_map.keys())
            for feature, value_map in first_class_dict.items()
            if feature != "__prior__"
        }

    def to_dict(self) -> Dict:
        """
        Returns the underlying model data as a dictionary for serialization.
        """
        return self._model
