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
    min_accuracy: float = 0.8,
) -> tuple[ClassifierService, dict]:
    """
    בונה את כל הצינור: טעינת נתונים, אימון מודל, הערכת דיוק
    ומחזיר גם את ClassifierService וגם את המודל הגולמי.
    """
    logger.info("Starting full model preparation…")

    # 1. טעינת נתונים וחלוקה
    data_handler = DataHandler(data_path=file_path)
    train_df, test_df = data_handler.get_split_data_as_dicts(
        target_col=target_col
    )

    # 2. בניית מודל Naive Bayes
    model_builder = NaiveBayesModelBuilder(alpha=1.0)
    trained_model = model_builder.build_model(train_df, target_col=target_col)

    # 3. עטיפת המודל ב-ClassifierService
    classifier = ClassifierService(model_artifact=trained_model)

    # 4. הערכת הביצועים
    evaluator = ModelEvaluatorService(classifier=classifier)
    list_test_data = DataHandler.get_data_as_list_of_dicts(test_df)
    accuracy_report = evaluator.run_evaluation(
        test_data=list_test_data, target_col=target_col
    )

    # הדפסה למסך (או ל-log) של תוצאות ההערכה
    display_accuracy_report(accuracy_report)

    # 5. בדיקת סף דיוק מינימלי
    if accuracy_report["accuracy"] < min_accuracy:
        raise RuntimeError(
            f"Model accuracy too low: {accuracy_report['accuracy']:.2%}"
        )

    return classifier, trained_model


def extract_expected_features(model: dict) -> dict[str, list[str]]:
    first_class_dict = next(iter(model.values()))
    return {
        feature: sorted(value_map.keys())
        for feature, value_map in first_class_dict.items()
        if feature != "__prior__"
    }
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
