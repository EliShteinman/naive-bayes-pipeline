# app/__init__.py
"""
This package contains all the modules for the Naive Bayes classifier pipeline.

It includes components for data handling, cleaning, model building,
evaluation, and the classifier service itself. It exposes the main pipeline
function and the core service classes for external use.
"""

from .classifier import ClassifierService
from .data_cleaner import DataCleaner
from .data_handler import DataHandler
from .data_splitter import DataSplitter
from .logger_config import get_logger
from .model_artifact import IModelArtifact, NaiveBayesDictArtifact
from .model_evaluator import ModelEvaluatorService
from .naive_bayes_model_builder import NaiveBayesModelBuilder

# Defines the public API of the 'app' package.
# When a user does 'from app import *', only these names will be imported.
__all__ = [
    # Core Services & Pipeline
    "ClassifierService",
    "ModelEvaluatorService",
    # Data Handling Components
    "DataHandler",
    "DataCleaner",
    "DataSplitter",
    # Model Building Components
    "NaiveBayesModelBuilder",
    # Model Artifact Interface
    "IModelArtifact",
    "NaiveBayesDictArtifact",
    # Logger Configuration
    "get_logger",
]
