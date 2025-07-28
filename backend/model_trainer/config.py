# model_trainer/config.py
import os

from dotenv import load_dotenv

load_dotenv()

# --- Configuration Loading ---
FILE_PATH = os.getenv("DATA_FILE_PATH")
TARGET_COL = os.getenv("TARGET_COL")
POS_LABEL = os.getenv("POS_LABEL")
TEST_SIZE = float(os.getenv("TEST_SIZE", 0.3))
RANDOM_STATE = int(os.getenv("RANDOM_STATE", 42))
VALIDATE_TEST_SET = os.getenv("VALIDATE_TEST_SET", "true").lower() == "true"
MIN_ACCURACY = float(os.getenv("MIN_ACCURACY", 0.8))
COLUMNS_TO_DROP = os.getenv("COLUMNS_TO_DROP", "").split(",")
ALPHA = float(os.getenv("ALPHA", 1.0))
