import redis
import json
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from datetime import datetime

def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['daily_spread'] = df['High'] - df['Low']
    df['rolling_mean'] = df['Close'].rolling(window=20).mean()
    df['rolling_std'] = df['Close'].rolling(window=20).std()
    df['z_score'] = (df['Close'] - df['rolling_mean']) / df['rolling_std']
    df['price_change_pct'] = df['Close'].pct_change()
    df.dropna(inplace=True)
    return df

def train_detector(df: pd.DataFrame) -> IsolationForest:
    features = df[['daily_spread', 'z_score', 'Volume','price_change_pct']]
    model = IsolationForest(contamination=0.03, random_state=42)
    model.fit(features)
    return model

def detect_anomalies(model: IsolationForest, df: pd.DataFrame) -> pd.DataFrame:
    features = df[['daily_spread', 'z_score', 'Volume','price_change_pct']]
    df = df.copy()
    df['anomaly_score'] = model.decision_function(features)
    df['is_anomaly'] = model.predict(features)
    df['is_anomaly'] = df['is_anomaly'].map({1: False, -1: True})
    return df

def log_anomalies(df: pd.DataFrame):
    anomalies = df[df['is_anomaly'] == True]
    with open('logs/anomalies.log', 'a') as f:
        for timestamp, row in anomalies.iterrows():
            log_entry = f"{datetime.now()} | {timestamp} | close={row['Close']:.2f} | spread={row['daily_spread']:.2f} | z_score={row['z_score']:.2f} | score={row['anomaly_score']:.4f} | change_pct={row['price_change_pct']:.4f}\n"
            f.write(log_entry)
            print(f"ANOMALY DETECTED: {log_entry}")

if __name__ == "__main__":
    from data.fetcher import fetch_historical_data
    
    df = fetch_historical_data("AAPL", period="2y")
    df = calculate_features(df)
    model = train_detector(df)
    df = detect_anomalies(model, df)
    log_anomalies(df)
    print(f"\nTotal anomalies detected: {df['is_anomaly'].sum()} out of {len(df)} trading days")