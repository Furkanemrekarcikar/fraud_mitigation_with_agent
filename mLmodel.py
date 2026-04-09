from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import joblib
import numpy as np
import pandas as pd


def feature_engineering(df):
    df['amount_log'] = np.log1p(df['amount'])
    df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
    df['high_amount'] = (df['amount'] > 1000).astype(int)

    if 'timestamp' in df.columns:
        df = df.drop(columns=['timestamp'])

    return df

def train_model(df):
    X = df.drop(columns=['label'])
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1
    )

    model.fit(X_train, y_train)
    joblib.dump(model, "fraud_model.pkl")

    return model

if __name__ == "__main__":
    df = pd.read_csv("creditcard.csv")



    df = df.rename(columns={
        "Amount": "amount",
        "Class": "label",
        "Time": "timestamp"
    })


    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')

    df = feature_engineering(df)
    model = train_model(df)
    print("Model eğitildi ve kaydedildi.")
