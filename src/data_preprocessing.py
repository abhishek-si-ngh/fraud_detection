import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(path='../data/dataset.csv'):
    df = pd.read_csv(path)
    return df

def preprocess_data(df):
    # Extract features
    X = df[['TX_AMOUNT', 'CUSTOMER_ID', 'TERMINAL_ID']]  # Use simple features
    y = df['TX_FRAUD']
    
    # Encode categorical features
    X = pd.get_dummies(X, columns=['CUSTOMER_ID', 'TERMINAL_ID'])
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test
