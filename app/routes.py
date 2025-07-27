import pandas as pd
import joblib
import csv
from datetime import datetime
from flask import Blueprint, request, render_template, redirect, url_for
from src.train_ml import train_model  # assumes this exists

main = Blueprint("main", __name__)

# Load ML model
def load_model():
    loaded = joblib.load("models/fraud_model_ml.pkl")
    return loaded["model"], loaded["features"]

model, feature_cols = load_model()

@main.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        customer_id = request.form["customer_id"]
        terminal_id = request.form["terminal_id"]
        tx_amount = float(request.form["tx_amount"])

        # Create input DataFrame
        df = pd.DataFrame([{
            "CUSTOMER_ID": customer_id,
            "TERMINAL_ID": terminal_id,
            "TX_AMOUNT": tx_amount,
            "TX_DATETIME": pd.Timestamp.now()
        }])
        df["TX_DOW"] = df["TX_DATETIME"].dt.dayofweek
        df["TX_HOUR"] = df["TX_DATETIME"].dt.hour
        df["IS_HIGH_AMOUNT"] = (df["TX_AMOUNT"] > 220).astype(int)
        df["CUST_FREQ"] = 1  # default if unknown
        df["TERM_FREQ"] = 1

        X_input = df[feature_cols]
        pred = model.predict(X_input)[0]
        prediction = "🚨 FRAUDULENT" if pred == 1 else "✅ LEGITIMATE"

        # Log prediction
        log_path = "data/prediction_logs.csv"
        log_entry = [
            datetime.now().isoformat(),
            customer_id,
            terminal_id,
            tx_amount,
            pred
        ]

        # Write header if file doesn't exist
        try:
            with open(log_path, "x", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "customer_id", "terminal_id", "tx_amount", "prediction"])
        except FileExistsError:
            pass  # file already exists

        # Append prediction
        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(log_entry)

    return render_template("index.html", prediction=prediction)

@main.route("/retrain", methods=["POST"])
def retrain():
    train_model()
    global model, feature_cols
    model, feature_cols = load_model()
    return redirect(url_for("main.index"))
