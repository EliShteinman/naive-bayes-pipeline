from typing import Dict, Any
from nb_classifier.data_handler import DataHandler
from nb_classifier.naive_bayes_model_builder import NaiveBayesModelBuilder
from nb_classifier.classifier import ClassifierService
from nb_classifier.model_evaluator import ModelEvaluatorService
from nb_classifier.logger_config import get_logger

logger = get_logger(__name__)

def display_prediction(sample: Dict, prediction: Any):
    print(f"\nFor sample: {sample}")
    print(f"The model predicts: {prediction}")


def display_accuracy_report(report: Dict):
    print("\n--- Model Evaluation Report ---")
    print(f"Accuracy: {report['accuracy']:.2%}")
    print(f"Total samples tested: {report['total_samples']}")
    print(f"Correctly classified: {report['correct_predictions']}")
    print(f"Incorrectly classified: {report['incorrect_predictions']}")


FILE_PATH = '/Users/lyhwstynmn/פרוייקטים/python/naive-bayes-pipeline/data/mushroom_decoded.csv'
TARGET_COL = 'poisonous'

def prepare_model_pipeline(
    file_path: str = FILE_PATH,
    target_col: str = TARGET_COL,
    min_accuracy: float = 0.8
) -> tuple[ClassifierService, dict]:
    logger.info("Starting full model preparation...")

    data_handler = DataHandler(data_path=file_path)
    train_data, test_data = data_handler.get_split_data_as_dicts(target_col=target_col)

    model_builder = NaiveBayesModelBuilder(alpha=1.0)
    trained_model = model_builder.build_model(train_data, target_col=target_col)

    classifier = ClassifierService(model_artifact=trained_model)

    evaluator = ModelEvaluatorService(classifier=classifier)
    list_test_data = data_handler.get_data_as_list_of_dicts(test_data)
    accuracy_report = evaluator.run_evaluation(test_data=list_test_data, target_col=target_col)

    display_accuracy_report(accuracy_report)

    if accuracy_report["accuracy"] < min_accuracy:
        raise RuntimeError(f"Model accuracy too low: {accuracy_report['accuracy']:.2%}")

    return classifier, trained_model


if __name__ == "__main__":
    logger.info("1. Preparing data...")
    data_handler = DataHandler(data_path=FILE_PATH)
    train_data, test_data = data_handler.get_split_data_as_dicts(target_col=TARGET_COL)
    logger.info("Data preparation complete.")

    logger.info("\n2. Building Naive Bayes model...")
    model_builder = NaiveBayesModelBuilder(alpha=1.0)
    trained_model = model_builder.build_model(train_data, target_col=TARGET_COL)
    logger.debug(trained_model)  # הדפסת המודל כדי לוודא שהוא נבנה כראוי
    logger.info("Model built successfully.")


    logger.info("\n3. Evaluating model performance...")
    evaluator = ModelEvaluatorService(classifier=ClassifierService(model_artifact=trained_model))
    list_test_data = data_handler.get_data_as_list_of_dicts(test_data)
    accuracy_report = evaluator.run_evaluation(test_data=list_test_data, target_col=TARGET_COL)
    display_accuracy_report(accuracy_report)



    logger.info("\n4. Classify a single sample...")
    classifier = ClassifierService(model_artifact=trained_model)
