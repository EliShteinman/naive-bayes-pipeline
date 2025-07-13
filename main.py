from nb_classifier.data_handler import DataHandler
from nb_classifier.naive_bayes_model_builder import NaiveBayesModelBuilder
from nb_classifier.classifier import ClassifierService
from nb_classifier.model_evaluator import ModelEvaluatorService
from nb_classifier.application_manager import display_prediction, display_accuracy_report
from nb_classifier.logger_config import get_logger

logger = get_logger(__name__)
FILE_PATH = '/Users/lyhwstynmn/פרוייקטים/python/naive-bayes-pipeline/data/mushroom_decoded.csv'
TARGET_COL = 'poisonous'

if __name__ == "__main__":
    logger.info("1. Preparing data...")
    data_handler = DataHandler(data_path=FILE_PATH)
    train_data, test_data = data_handler.get_split_data_as_dicts(target_col=TARGET_COL)
    logger.info("Data preparation complete.")

    logger.info("\n2. Building Naive Bayes model...")
    model_builder = NaiveBayesModelBuilder(alpha=1.0)
    trained_model = model_builder.build_model(train_data, target_col=TARGET_COL)
    logger.info("Model built successfully.")


    logger.info("\n3. Evaluating model performance...")
    evaluator = ModelEvaluatorService(classifier=ClassifierService(model_artifact=trained_model))
    list_test_data = data_handler.get_data_as_list_of_dicts(test_data)
    accuracy_report = evaluator.run_evaluation(test_data=list_test_data, target_col=TARGET_COL)
    logger.info(accuracy_report)



    logger.info("\n4. Classify a single sample...")
    classifier = ClassifierService(model_artifact=trained_model)