# Fraud Transaction Detection App 🚨

A full-stack machine learning application to detect fraudulent transactions using a trained ML model (Gradient Boosting). This app is built using Flask, deployed on Render, and includes a responsive web UI for live predictions.

## 🔍 Features

- 🧠 Real ML model (Gradient Boosting via scikit-learn)
- 🖥️ Responsive web interface (HTML + CSS)
- 📊 Logs all predictions to `data/prediction_logs.csv`
- 🔁 Retrain model on demand with a single button
- ☁️ Deployed for free using [Render](https://render.com)

## 📁 Project Structure

```
fraud_detection/
│
├── app/                   # Flask app code
│   ├── __init__.py
│   ├── routes.py
│   └── templates/
│       └── index.html
│
├── data/                  # Raw data and prediction logs
│   ├── transactions.csv
│   └── prediction_logs.csv (generated during use)
│
├── models/                # Trained model
│   └── fraud_model_ml.pkl
│
├── notebooks/             # EDA notebook (optional)
│   └── EDA.ipynb
│
├── src/                   # Custom training, preprocessing scripts
│   ├── data_preprocessing.py
│   ├── model_training.py
│   ├── train_ml.py
│   └── ...
│
├── main.py                # Entry-point (can trigger training)
├── wsgi.py                # For Render deployment
├── requirements.txt       # Python dependencies
├── Procfile               # Tells Render how to run the app
└── .gitignore
```

## 🚀 Live App

You can access the deployed app here: _[your Render URL]_

## 🛠️ How to Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/your-username/fraud-detection-app.git
cd fraud-detection-app

# 2. Create virtual environment and install dependencies
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt

# 3. Train the model (if not already present)
python src/train_ml.py

# 4. Run the app
python main.py
# or
gunicorn wsgi:app
```

## 🧪 Example Inputs

| CUSTOMER_ID | TERMINAL_ID | TX_AMOUNT |
|-------------|-------------|-----------|
| 1234        | 9876        | 299.50    |
| 4321        | 1234        | 49.99     |

## 📊 Logs

Every prediction made via the UI is appended to `data/prediction_logs.csv` like:

```
timestamp,customer_id,terminal_id,tx_amount,prediction
2025-07-26T12:01:22,1234,9876,299.5,1
```

## 📦 Deployment Notes

- Use `gunicorn wsgi:app` as the start command on Render
- Make sure `models/fraud_model_ml.pkl` is committed or generated during build

## 📄 License

MIT License.