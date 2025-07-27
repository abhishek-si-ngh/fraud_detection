import pandas as pd
import joblib
from flask import Blueprint, request, render_template

main = Blueprint("main", __name__)

# Load ML model
loaded = joblib.load("models/fraud_model_ml.pkl")
model = loaded["model"]
feature_cols = loaded["features"]

@main.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        customer_id = request.form["customer_id"]
        terminal_id = request.form["terminal_id"]
        tx_amount = float(request.form["tx_amount"])

        # Create input DataFrame and engineer features
        df = pd.DataFrame([{
            "CUSTOMER_ID": customer_id,
            "TERMINAL_ID": terminal_id,
            "TX_AMOUNT": tx_amount,
            "TX_DATETIME": pd.Timestamp.now()
        }])
        df["TX_DOW"] = df["TX_DATETIME"].dt.dayofweek
        df["TX_HOUR"] = df["TX_DATETIME"].dt.hour
        df["IS_HIGH_AMOUNT"] = (df["TX_AMOUNT"] > 220).astype(int)
        df["CUST_FREQ"] = 1  # assume 1 if unknown
        df["TERM_FREQ"] = 1

        X_input = df[feature_cols]

        pred = model.predict(X_input)[0]
        prediction = "🚨 FRAUDULENT" if pred == 1 else "✅ LEGITIMATE"

    return render_template("index.html", prediction=prediction)
