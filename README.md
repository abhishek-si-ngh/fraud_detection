# Fraud Transaction Detection App 🚨

A full-stack machine learning web application to detect fraudulent transactions using a trained Gradient Boosting model. Built with Flask and deployed for free on Render.

🔗 **Live App**: [fraud-detection-opba.onrender.com](https://fraud-detection-opba.onrender.com)

## 🔍 Features

- 🧠 ML model (Gradient Boosting Classifier via scikit-learn)
- 💻 Clean web interface built with Flask + HTML + CSS
- 📱 Mobile-responsive UI design
- 📊 Logs all predictions to `data/prediction_logs.csv`
- 🔁 One-click retraining of the model via UI button
- ☁️ Fully deployed on [Render](https://render.com)

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
│   └── prediction_logs.csv (auto-generated)
│
├── models/                # Trained ML model file
│   └── fraud_model_ml.pkl
│
├── notebooks/             # Exploratory Data Analysis
│   └── EDA.ipynb
│
├── src/                   # Custom training & utilities
│   ├── data_preprocessing.py
│   ├── model_training.py
│   ├── train_ml.py
│   └── ...
│
├── main.py                # CLI entry to train model
├── wsgi.py                # Render entry point
├── requirements.txt       # Python packages
├── Procfile               # Tells Render how to start app
└── .gitignore             # Files to ignore in Git
```

## 🚀 Running the App Locally

```bash
# Clone the repo
git clone https://github.com/your-username/fraud-detection-app.git
cd fraud-detection-app

# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate   # on Windows
# or
source .venv/bin/activate  # on Mac/Linux

# Install dependencies
pip install -r requirements.txt

# (Optional) Train ML model
python src/train_ml.py

# Run app locally
python main.py
# or
gunicorn wsgi:app
```

## 🧪 Example Input

| CUSTOMER_ID | TERMINAL_ID | TX_AMOUNT |
|-------------|-------------|-----------|
| 1234        | 9876        | 299.50    |
| 4321        | 1234        | 49.99     |

## 📊 Prediction Logs

Every prediction made through the UI is appended to:

```bash
data/prediction_logs.csv
```

Format:
```
timestamp,customer_id,terminal_id,tx_amount,prediction
2025-07-26T12:01:22,1234,9876,299.5,1
```

## 💡 Future Improvements

- Add confidence score (% fraud probability)
- User authentication for admins
- Display logged predictions in table on UI

## 📄 License

MIT License.