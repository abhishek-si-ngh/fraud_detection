import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score

def engineer_features(df):
    df["TX_DATETIME"] = pd.to_datetime(df["TX_DATETIME"])
    df["TX_DOW"] = df["TX_DATETIME"].dt.dayofweek
    df["TX_HOUR"] = df["TX_DATETIME"].dt.hour
    df["IS_HIGH_AMOUNT"] = (df["TX_AMOUNT"] > 220).astype(int)
    df["CUST_FREQ"] = df["CUSTOMER_ID"].map(df["CUSTOMER_ID"].value_counts())
    df["TERM_FREQ"] = df["TERMINAL_ID"].map(df["TERMINAL_ID"].value_counts())

    X = df[["TX_AMOUNT", "TX_DOW", "TX_HOUR", "IS_HIGH_AMOUNT", "CUST_FREQ", "TERM_FREQ"]]
    y = df["TX_FRAUD"].astype(int)
    return X, y

def train_model():
    df = pd.read_csv("data/transactions.csv")
    X, y = engineer_features(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    clf = HistGradientBoostingClassifier(random_state=42)
    clf.fit(X_train, y_train)

    # Save model
    model_path = Path("models/fraud_model_ml.pkl")
    joblib.dump({"model": clf, "features": list(X.columns)}, model_path)
    print(f"✅ Model trained and saved to {model_path}")

    # Evaluation (optional)
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]
    print(classification_report(y_test, y_pred))
    print("ROC AUC:", roc_auc_score(y_test, y_proba))
    print("Average Precision:", average_precision_score(y_test, y_proba))

if __name__ == "__main__":
    train_model()
