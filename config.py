"""
Configuration module for Stock Anomaly Detector
Loads settings from environment variables with sensible defaults
"""
import os
from pathlib import Path


class Config:
    """Base configuration"""
    
    # Data Fetching
    STOCK_TICKER = os.getenv("STOCK_TICKER", "AAPL")
    DATA_PERIOD = os.getenv("DATA_PERIOD", "2y")
    
    # Redis Configuration
    REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
    REDIS_STREAM_NAME = os.getenv("REDIS_STREAM_NAME", "stock_stream")
    
    # Stream Producer
    STREAM_DELAY = float(os.getenv("STREAM_DELAY", 0.3))
    
    # Anomaly Detection
    ISOLATION_FOREST_CONTAMINATION = float(os.getenv("ISOLATION_FOREST_CONTAMINATION", 0.03))
    ROLLING_WINDOW = int(os.getenv("ROLLING_WINDOW", 20))
    
    # API Server
    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("API_PORT", 8000))
    API_RELOAD = os.getenv("API_RELOAD", "true").lower() == "true"
    
    # Logging
    LOG_DIR = os.getenv("LOG_DIR", "logs")
    LOG_FILE = os.path.join(LOG_DIR, os.getenv("LOG_FILE", "anomalies.log"))
    
    @staticmethod
    def ensure_log_dir():
        """Create log directory if it doesn't exist"""
        Path(Config.LOG_DIR).mkdir(exist_ok=True)


if __name__ == "__main__":
    print("Configuration loaded successfully")
    print(f"Stock Ticker: {Config.STOCK_TICKER}")
    print(f"Redis: {Config.REDIS_HOST}:{Config.REDIS_PORT}")
    print(f"Contamination: {Config.ISOLATION_FOREST_CONTAMINATION}")
