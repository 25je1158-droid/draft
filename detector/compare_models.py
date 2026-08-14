import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from datetime import datetime
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.fetcher import fetch_historical_data
from detector.anomaly import calculate_features

def load_known_events(csv_path: str) -> pd.DataFrame:
    events = pd.read_csv(csv_path)
    events['date'] = pd.to_datetime(events['date']).dt.tz_localize(None)
    return events

def build_sample(df: pd.DataFrame, known_dates: list, n_random: int = 34) -> pd.DataFrame:
    df.index = df.index.tz_localize(None)
    known_rows = df[df.index.normalize().isin(known_dates)]
    remaining = df[~df.index.normalize().isin(known_dates)]
    random_rows = remaining.sample(n=min(n_random, len(remaining)), random_state=42)
    sample = pd.concat([known_rows, random_rows]).sort_index()
    return sample

def zscore_detect(df: pd.DataFrame, threshold: float = 2.0) -> pd.Series:
    return (abs(df['z_score']) > threshold) | (abs(df['price_change_pct']) > threshold)

def lof_detect(df: pd.DataFrame) -> pd.Series:
    features = df[['daily_spread', 'z_score', 'Volume', 'price_change_pct']]
    lof = LocalOutlierFactor(n_neighbors=5, contamination=0.03)
    preds = lof.fit_predict(features)
    return pd.Series(preds == -1, index=df.index)

def isolation_forest_detect(df: pd.DataFrame) -> pd.Series:
    features = df[['daily_spread', 'z_score', 'Volume', 'price_change_pct']]
    model = IsolationForest(contamination=0.03, random_state=42)
    preds = model.fit_predict(features)
    return pd.Series(preds == -1, index=df.index)

def evaluate_models(sample: pd.DataFrame, known_dates: list):
    sample.index = sample.index.tz_localize(None) if sample.index.tzinfo else sample.index

    zscore_flags = zscore_detect(sample)
    lof_flags = lof_detect(sample)
    if_flags = isolation_forest_detect(sample)

    print("\n" + "="*60)
    print("MODEL COMPARISON REPORT")
    print("="*60)
    print(f"Sample size: {len(sample)} trading days")
    print(f"Known anomaly events: {len(known_dates)}")
    print("="*60)

    results = {}
    for name, flags in [("Z-Score", zscore_flags), ("LOF", lof_flags), ("Isolation Forest", if_flags)]:
        detected = []
        missed = []
        for date in known_dates:
            date = pd.Timestamp(date).normalize()
            match = sample[sample.index.normalize() == date]
            if not match.empty and flags[match.index[0]]:
                detected.append(date)
            else:
                missed.append(date)
        recall = len(detected) / len(known_dates) * 100
        results[name] = {
            'detected': len(detected),
            'missed': len(missed),
            'recall': recall,
            'total_flagged': flags.sum()
        }
        print(f"\n{name}:")
        print(f"  Known events detected : {len(detected)}/{len(known_dates)} ({recall:.1f}% recall)")
        print(f"  Total flags in sample : {flags.sum()}")
        print(f"  Missed events         : {[str(d.date()) for d in missed]}")

    print("\n" + "="*60)
    winner = max(results, key=lambda x: results[x]['recall'])
    print(f"WINNER: {winner} with {results[winner]['recall']:.1f}% recall on known events")
    print(f"Selected for full dataset analysis.")
    print("="*60)
    return winner

if __name__ == "__main__":
    print("Fetching data...")
    df = fetch_historical_data("AAPL", period="2y")
    df = calculate_features(df)

    print("Loading known events...")
    events = load_known_events("data/known_events.csv")
    known_dates = events['date'].tolist()

    print("Building sample dataset...")
    sample = build_sample(df, known_dates, n_random=34)
    print(f"Sample built: {len(sample)} rows ({len(events)} known events + random normal days)")

    winner = evaluate_models(sample, known_dates)

    print(f"\nRunning {winner} on full dataset ({len(df)} trading days)...")
    if winner == "Isolation Forest":
        from detector.anomaly import train_detector, detect_anomalies, log_anomalies
        model = train_detector(df)
        result = detect_anomalies(model, df)
        log_anomalies(result)
        print(f"Total anomalies detected: {result['is_anomaly'].sum()} out of {len(df)} trading days")