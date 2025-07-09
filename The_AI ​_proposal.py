"""
Mushroom Classification and Evaluation Script using an Optimized Naive Bayes Classifier.

This script demonstrates a full, efficient, and robust workflow:

1.  Loads and pre-processes the mushroom dataset using efficient pandas chaining.
2.  Splits data into training and testing sets, validating the test set with
    vectorized operations.
3.  Trains a Naive Bayes model using fast, vectorized counting (`groupby`).
4.  Converts counts to log-weights using standard, unconditional Laplace smoothing.
5.  Evaluates the model's accuracy on the unseen test data.
6.  Robustly and interactively prompts a user for a new sample, handling input errors.
7.  Classifies the sample, gracefully ignoring any unseen values, and prints the result.

To run the script, execute from the command line:
    python your_optimized_script_name.py
"""

from copy import deepcopy
import pandas as pd
from math import log
from sklearn.model_selection import train_test_split
from typing import Tuple, Any


def load_and_clean_mushroom_data(
    filepath: str = "data/mushroom_decoded.csv",
) -> pd.DataFrame:
    """
    Loads and pre-processes a dataset from a specified CSV file.

    Designed for the mushroom dataset, this function performs the following:
    1. Reads the data from the given `filepath`.
    2. Drops the 'stalk-root' column, which is known for missing values.
    3. Removes any columns with no variance (fewer than two unique values).

    Args:
        filepath (str, optional): The path to the CSV file.
                                  Defaults to 'data/mushroom_decoded.csv'.

    Returns:
        pd.DataFrame: A cleaned and pre-processed DataFrame ready for analysis.
    """
    cleaned_data = (
        pd.read_csv(filepath)
        .drop(columns=["stalk-root"], errors="ignore")
        .loc[:, lambda df: df.nunique() > 1]
    )
    return cleaned_data


def split_and_validate_data_optimized(
    data: pd.DataFrame, target_col: str, test_size: float = 0.3, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits data and efficiently validates the test set using vectorized operations.

    This function first performs a stratified split. It then efficiently
    identifies and removes rows from the test set that contain categorical
    values not seen in the training set.

    Args:
        data (pd.DataFrame): The full dataset to be split.
        target_col (str): The name of the target variable column for stratification.
        test_size (float, optional): The proportion for the test split. Defaults to 0.3.
        random_state (int, optional): Seed for reproducibility. Defaults to 42.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the training and validated test sets.
    """
    train_df, test_df = train_test_split(
        data, test_size=test_size, random_state=random_state, stratify=data[target_col]
    )

    feature_cols = [col for col in data.columns if col != target_col]
    train_uniques = {col: set(train_df[col].unique()) for col in feature_cols}

    mask_unseen = (
        ~test_df[feature_cols]
        .apply(lambda col: col.isin(train_uniques[col.name]))
        .all(axis=1)
    )

    indices_to_drop = test_df[mask_unseen].index

    if not indices_to_drop.empty:
        print(
            f"הוסרו {len(indices_to_drop)} שורות מסט הבדיקה בגלל ערכים שלא הופיעו באימון."
        )
        test_df = test_df.drop(index=indices_to_drop)

    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def get_model_optimized(data: pd.DataFrame, target_col: str) -> dict:
    """
    Builds a "model" for a Naive Bayes classifier efficiently using pandas.

    Args:
        data (pd.DataFrame): The DataFrame containing the training data.
        target_col (str): The name of the target variable column.

    Returns:
        dict: A dictionary containing the model's components.
    """
    feature_cols = [col for col in data.columns if col != target_col]
    target_counts = data[target_col].value_counts().to_dict()

    unique_value_dicts = {
        col: {val: 0 for val in data[col].unique()} for col in feature_cols
    }
    likelihoods = {
        target_val: deepcopy(unique_value_dicts)
        for target_val in data[target_col].unique()
    }

    grouped_by_target = data.groupby(target_col)
    for col in feature_cols:
        counts = grouped_by_target[col].value_counts().unstack(fill_value=0)
        for target_val, value_counts_series in counts.iterrows():
            likelihoods[target_val][col].update(value_counts_series.to_dict())

    return {
        "likelihoods": likelihoods,
        "target_counts": target_counts,
        "total_count": len(data),
    }


def convert_counts_to_weights_optimized(model: dict, alpha: float = 1.0) -> dict:
    """
    Converts a count-based model to log-probabilities using standard Laplace smoothing.

    Args:
        model (dict): A dictionary from `get_model` with counts.
        alpha (float, optional): The smoothing parameter. Defaults to 1.0.

    Returns:
        dict: A new dictionary of weights (log-probabilities).
    """
    weights = {}
    for target_val, feature_counts in model["likelihoods"].items():
        weights[target_val] = {}
        log_prior = log(model["target_counts"][target_val] / model["total_count"])
        weights[target_val]["__prior__"] = log_prior

        for feature_name, value_counts in feature_counts.items():
            total_feature_count = sum(value_counts.values())
            num_unique_values = len(value_counts)
            denominator = total_feature_count + alpha * num_unique_values
            weights[target_val][feature_name] = {
                feature_val: log((count + alpha) / denominator)
                for feature_val, count in value_counts.items()
            }
    return weights


def get_user_sample_robust(data: pd.DataFrame, target_col: str) -> dict:
    """
    Interactively and robustly prompts the user to build a sample data point.

    Args:
        data (pd.DataFrame): The DataFrame to source feature columns and values from.
        target_col (str): The name of the target column to be excluded.

    Returns:
        dict: A dictionary representing the data sample created by the user.
    """
    sample = {}
    feature_cols = [col for col in data.columns if col != target_col]

    for col in feature_cols:
        options = list(data[col].dropna().unique())
        print(f"\nSelect a value for the column: {col}")
        for i, opt in enumerate(options):
            print(f"  {i + 1}. {opt}")

        while True:
            try:
                choice_str = input(f"Enter a number between 1 and {len(options)}: ")
                choice_idx = int(choice_str) - 1
                if 0 <= choice_idx < len(options):
                    sample[col] = options[choice_idx]
                    break
                else:
                    print(
                        f"Invalid choice. Please enter a number between 1 and {len(options)}."
                    )
            except ValueError:
                print("Invalid input. Please enter a number.")
    return sample


def classify_sample(sample: dict, weights: dict) -> Any:
    """
    Classifies a sample, gracefully handling unseen values.

    Args:
        sample (dict): A dictionary representing the data sample to classify.
        weights (dict): A dictionary of pre-calculated log-probabilities.

    Returns:
        Any: The predicted class label with the highest log-probability.
    """
    best_class = None
    best_log_prob = float("-inf")

    for target_value in weights.keys():
        log_prob = weights[target_value]["__prior__"]
        for feature, value in sample.items():
            feature_weights = weights[target_value].get(feature, {})
            log_prob += feature_weights.get(value, 0.0)

        if log_prob > best_log_prob:
            best_log_prob = log_prob
            best_class = target_value
    return best_class


def evaluate_model(test_df: pd.DataFrame, target_col: str, weights: dict) -> float:
    """Calculates the accuracy of the model on a test set."""
    X_test = test_df.drop(columns=target_col)
    y_test = test_df[target_col]
    correct_predictions = sum(
        1
        for i in range(len(X_test))
        if classify_sample(X_test.iloc[i].to_dict(), weights) == y_test.iloc[i]
    )
    return correct_predictions / len(test_df) if len(test_df) > 0 else 0.0


if __name__ == "__main__":
    # --- 1. Data Loading and Preparation ---
    print("Loading and preparing data...")
    mushroom_data = load_and_clean_mushroom_data()
    train_data, test_data = split_and_validate_data_optimized(
        mushroom_data, target_col="poisonous"
    )

    # --- 2. Model Training ---
    print("Training the Naive Bayes model...")
    mushroom_model = get_model_optimized(train_data, "poisonous")
    mushroom_weights = convert_counts_to_weights_optimized(mushroom_model)

    # --- 3. Model Evaluation ---
    print("Evaluating model performance on the test set...")
    if not test_data.empty:
        accuracy = evaluate_model(test_data, "poisonous", mushroom_weights)
        print(f"--> Model Accuracy on Test Set: {accuracy:.2%}\n")
    else:
        print("--> Test set is empty, cannot evaluate model.\n")

    # --- 4. Interactive Classification ---
    print("--- Now, let's classify a new mushroom ---")
    user_sample = get_user_sample_robust(mushroom_data, target_col="poisonous")
    prediction = classify_sample(user_sample, mushroom_weights)

    print("\n-------------------------------------------")
    print(f"Your sample: {user_sample}")
    print(f"The model predicts the mushroom is: {prediction}")
    print("-------------------------------------------")
