"""
Improved anomaly detection module with error handling and logging
"""
import redis
import json
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from datetime import datetime
import logging

from config import Config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate statistical features for anomaly detection
    
    Features:
    - daily_spread: High - Low (volatility within the day)
    - rolling_mean: 20-day rolling average
    - rolling_std: 20-day rolling standard deviation
    - z_score: Standardized deviation from rolling mean
    - price_change_pct: Daily percentage change
    
    Args:
        df: DataFrame with OHLCV data
        
    Returns:
        DataFrame with engineered features, NaN rows removed
    """
    try:
        df = df.copy()
        
        # Validate input
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Calculate features
        df['daily_spread'] = df['High'] - df['Low']
        df['rolling_mean'] = df['Close'].rolling(window=Config.ROLLING_WINDOW).mean()
        df['rolling_std'] = df['Close'].rolling(window=Config.ROLLING_WINDOW).std()
        df['z_score'] = (df['Close'] - df['rolling_mean']) / df['rolling_std']
        df['price_change_pct'] = df['Close'].pct_change()
        
        initial_len = len(df)
        df.dropna(inplace=True)
        dropped = initial_len - len(df)
        
        if dropped > 0:
            logger.info(f"Dropped {dropped} rows due to NaN values (rolling window)")
        
        return df
    
    except Exception as e:
        logger.error(f"Error calculating features: {str(e)}")
        raise


def train_detector(df: pd.DataFrame) -> IsolationForest:
    """
    Train Isolation Forest model for anomaly detection
    
    Args:
        df: DataFrame with calculated features
        
    Returns:
        Trained IsolationForest model
    """
    try:
        features = df[['daily_spread', 'z_score', 'Volume', 'price_change_pct']]
        
        # Validate data
        if len(features) < 10:
            raise ValueError("Insufficient data for training (need at least 10 rows)")
        
        if features.isna().any().any():
            logger.warning("NaN values found in features, dropping rows")
            features = features.dropna()
        
        model = IsolationForest(
            contamination=Config.ISOLATION_FOREST_CONTAMINATION,
            random_state=42,
            n_estimators=100
        )
        model.fit(features)
        logger.info("Model trained successfully")
        return model
    
    except Exception as e:
        logger.error(f"Error training detector: {str(e)}")
        raise


def detect_anomalies(model: IsolationForest, df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect anomalies using trained model
    
    Args:
        model: Trained IsolationForest model
        df: DataFrame with features
        
    Returns:
        DataFrame with anomaly_score and is_anomaly columns
    """
    try:
        features = df[['daily_spread', 'z_score', 'Volume', 'price_change_pct']]
        
        # Validate data
        if features.isna().any().any():
            logger.warning("NaN values found, dropping rows before prediction")
            mask = features.notna().all(axis=1)
            df = df[mask]
            features = features[mask]
        
        df = df.copy()
        df['anomaly_score'] = model.decision_function(features)
        predictions = model.predict(features)
        
        # Map predictions: 1 = normal, -1 = anomaly
        df['is_anomaly'] = predictions == -1
        
        anomaly_count = df['is_anomaly'].sum()
        logger.info(f"Detected {anomaly_count} anomalies out of {len(df)} records")
        
        return df
    
    except Exception as e:
        logger.error(f"Error detecting anomalies: {str(e)}")
        raise


def log_anomalies(df: pd.DataFrame) -> int:
    """
    Log detected anomalies to file
    
    Args:
        df: DataFrame with anomaly predictions
        
    Returns:
        Number of anomalies logged
    """
    try:
        Config.ensure_log_dir()
        anomalies = df[df['is_anomaly'] == True]
        
        if len(anomalies) == 0:
            logger.info("No anomalies to log")
            return 0
        
        try:
            with open(Config.LOG_FILE, 'w') as f:
                for timestamp, row in anomalies.iterrows():
                    log_entry = (
                        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
                        f"{timestamp.strftime('%Y-%m-%d')} | "
                        f"close={row['Close']:.2f} | "
                        f"spread={row['daily_spread']:.2f} | "
                        f"z_score={row['z_score']:.2f} | "
                        f"score={row['anomaly_score']:.4f} | "
                        f"change_pct={row['price_change_pct']:.4f}\n"
                    )
                    f.write(log_entry)
                    print(f"ANOMALY DETECTED: {log_entry.strip()}")
            
            logger.info(f"Logged {len(anomalies)} anomalies to {Config.LOG_FILE}")
            return len(anomalies)
        
        except IOError as e:
            logger.error(f"Failed to write to log file: {str(e)}")
            raise
    
    except Exception as e:
        logger.error(f"Error logging anomalies: {str(e)}")
        raise


def connect_redis(max_retries: int = 3) -> redis.Redis:
    """
    Connect to Redis with retry logic
    
    Args:
        max_retries: Number of connection attempts
        
    Returns:
        Redis connection object
        
    Raises:
        ConnectionError if all retries fail
    """
    for attempt in range(max_retries):
        try:
            r = redis.Redis(
                host=Config.REDIS_HOST,
                port=Config.REDIS_PORT,
                decode_responses=True,
                socket_connect_timeout=5
            )
            r.ping()
            logger.info(f"Connected to Redis at {Config.REDIS_HOST}:{Config.REDIS_PORT}")
            return r
        except redis.ConnectionError as e:
            logger.warning(f"Redis connection attempt {attempt + 1}/{max_retries} failed: {str(e)}")
            if attempt == max_retries - 1:
                raise ConnectionError(f"Failed to connect to Redis after {max_retries} attempts")
            continue


def read_stream(r: redis.Redis, stream_name: str, batch_size: int = 100) -> pd.DataFrame:
    """
    Read data from Redis stream and convert to DataFrame
    
    Args:
        r: Redis connection
        stream_name: Name of the stream
        batch_size: Number of records to read
        
    Returns:
        DataFrame with stream data
    """
    try:
        messages = r.xrange(stream_name, count=batch_size)
        
        if not messages:
            logger.warning(f"No messages found in stream: {stream_name}")
            return pd.DataFrame()
        
        data = []
        for msg_id, msg_data in messages:
            data.append(msg_data)
        
        df = pd.DataFrame(data)
        logger.info(f"Read {len(df)} records from stream")
        return df
    
    except Exception as e:
        logger.error(f"Error reading from Redis stream: {str(e)}")
        raise


if __name__ == "__main__":
    try:
        from data.fetcher import fetch_historical_data
        
        logger.info(f"Starting anomaly detection for {Config.STOCK_TICKER}")
        
        # Fetch data
        df = fetch_historical_data(Config.STOCK_TICKER, period=Config.DATA_PERIOD)
        logger.info(f"Fetched {len(df)} records")
        
        # Calculate features
        df = calculate_features(df)
        logger.info(f"Calculated features for {len(df)} records")
        
        # Train model
        model = train_detector(df)
        
        # Detect anomalies
        df = detect_anomalies(model, df)
        
        # Log anomalies
        count = log_anomalies(df)
        
        print(f"\n{'='*50}")
        print(f"Total anomalies detected: {df['is_anomaly'].sum()} out of {len(df)} trading days")
        print(f"Anomaly rate: {(df['is_anomaly'].sum() / len(df) * 100):.2f}%")
        print(f"{'='*50}")
        
    except Exception as e:
        logger.error(f"Fatal error in anomaly detection: {str(e)}")
        raise
