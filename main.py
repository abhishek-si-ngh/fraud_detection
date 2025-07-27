"""
Main runner: ensures data extracted, CSV built, loads baseline rule model (or trains ML).
"""

from pathlib import Path
import pandas as pd
import joblib

from src import data_utils
from src.rule_model import FraudRuleModel

ZIP_PATH = "dataset.zip"
EXTRACT_DIR = "data/raw_daily_pickles"
CSV_PATH = "data/transactions.csv"
RULE_MODEL_PATH = "models/fraud_model.pkl"

def prepare_data():
    # If CSV already exists, skip.
    if Path(CSV_PATH).exists():
        print("CSV already prepared.")
        return
    # Extract zip (assumes dataset.zip at project root)
    if not Path(EXTRACT_DIR).exists():
        print("Extracting zip...")
        data_utils.extract_zip(ZIP_PATH, "data")
    # Load + save
    print("Combining daily pickles...")
    df = data_utils.load_daily_pickles("data/data")  # adjust if layout differs
    data_utils.save_csv(df, CSV_PATH)
    print(f"Saved -> {CSV_PATH}")

def build_rule_model():
    print("Building rule-based baseline model...")
    df = pd.read_csv(CSV_PATH)
    model = FraudRuleModel.fit_from_dataframe(df)
    joblib.dump(model, RULE_MODEL_PATH)
    print(f"Saved rule model -> {RULE_MODEL_PATH}")

if __name__ == "__main__":
    prepare_data()
    build_rule_model()
