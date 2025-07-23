# backend/nb_classifier/__init__.py
"""
This package contains all the modules for the Naive Bayes classifier.

It includes components for data handling, cleaning, splitting, model building,
evaluation, and the classifier service itself.
"""

from .application_manager import prepare_model_pipeline
from .classifier import ClassifierService
from .data_cleaner import DataCleaner
from .data_handler import DataHandler
from .data_splitter import DataSplitter
from .df_utiles import DataFrameUtils
from .logger_config import get_logger
from .model_evaluator import ModelEvaluatorService
from .naive_bayes_model_builder import NaiveBayesModelBuilder

# You can define __all__ to control what 'from nb_classifier import *' imports
__all__ = [
    "get_logger",
    "DataHandler",
    "DataCleaner",
    "DataSplitter",
    "DataFrameUtils",
    "NaiveBayesModelBuilder",
    "ModelEvaluatorService",
    "ClassifierService",
    "prepare_model_pipeline",
]
