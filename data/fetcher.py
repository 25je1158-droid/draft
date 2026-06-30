"""
Improved data fetcher with error handling and logging
"""
import yfinance as yf
import pandas as pd
import time
import logging
from typing import Optional

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def fetch_historical_data(
    ticker: str,
    period: str = "2y",
    max_retries: int = 3,
    retry_delay: float = 1.0
) -> pd.DataFrame:
    """
    Fetch historical stock data from Yahoo Finance with retry logic
    
    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL')
        period: Data period (e.g., '1y', '2y', 'max')
        max_retries: Number of retry attempts
        retry_delay: Initial delay between retries in seconds
        
    Returns:
        DataFrame with OHLCV data
        
    Raises:
        ValueError: If ticker is invalid
        ConnectionError: If data fetch fails after retries
    """
    # Validate ticker
    if not ticker or not isinstance(ticker, str):
        raise ValueError("Invalid ticker symbol")
    
    ticker = ticker.upper().strip()
    
    # Valid period values
    valid_periods = ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max']
    if period not in valid_periods:
        logger.warning(f"Period '{period}' not standard, attempting anyway")
    
    for attempt in range(max_retries):
        try:
            logger.info(f"Fetching {ticker} data (period: {period}, attempt {attempt + 1}/{max_retries})")
            
            # Fetch data
            stock = yf.Ticker(ticker)
            df = stock.history(period=period)
            
            # Validate response
            if df.empty:
                raise ValueError(f"No data returned for ticker {ticker}")
            
            # Select and validate columns
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                raise ValueError(f"Missing columns in data: {missing_cols}")
            
            df = df[required_cols]
            
            # Remove NaN rows
            initial_len = len(df)
            df = df.dropna()
            dropped = initial_len - len(df)
            
            if dropped > 0:
                logger.warning(f"Dropped {dropped} rows with NaN values")
            
            # Validate data
            if len(df) == 0:
                raise ValueError(f"No valid data available for {ticker}")
            
            # Check for all-zero volumes
            if (df['Volume'] == 0).all():
                logger.warning(f"All volume values are zero for {ticker}")
            
            logger.info(f"✓ Successfully fetched {len(df)} records for {ticker}")
            logger.info(f"  Date range: {df.index[0].date()} to {df.index[-1].date()}")
            logger.info(f"  Price range: ${df['Close'].min():.2f} - ${df['Close'].max():.2f}")
            
            return df
        
        except (ValueError, KeyError) as e:
            # Don't retry for validation errors
            logger.error(f"Data validation error for {ticker}: {str(e)}")
            raise
        
        except Exception as e:
            wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
            
            logger.warning(
                f"Attempt {attempt + 1}/{max_retries} failed for {ticker}: {str(e)}"
            )
            
            if attempt < max_retries - 1:
                logger.info(f"Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise ConnectionError(
                    f"Failed to fetch data for {ticker} after {max_retries} attempts. "
                    f"Check ticker symbol and network connection."
                )


def fetch_multiple_tickers(
    tickers: list,
    period: str = "2y",
    max_retries: int = 3
) -> dict:
    """
    Fetch data for multiple tickers
    
    Args:
        tickers: List of ticker symbols
        period: Data period
        max_retries: Number of retries per ticker
        
    Returns:
        Dictionary with ticker as key and DataFrame as value
    """
    results = {}
    
    for ticker in tickers:
        try:
            logger.info(f"Fetching {ticker}...")
            results[ticker] = fetch_historical_data(ticker, period, max_retries)
        except Exception as e:
            logger.error(f"Failed to fetch {ticker}: {str(e)}")
            results[ticker] = None
    
    successful = sum(1 for v in results.values() if v is not None)
    logger.info(f"Successfully fetched {successful}/{len(tickers)} tickers")
    
    return results


def validate_data_quality(df: pd.DataFrame) -> dict:
    """
    Validate data quality and return metrics
    
    Args:
        df: DataFrame with OHLCV data
        
    Returns:
        Dictionary with quality metrics
    """
    metrics = {
        "total_records": len(df),
        "date_range": f"{df.index[0].date()} to {df.index[-1].date()}",
        "missing_values": df.isna().sum().sum(),
        "zero_volumes": (df['Volume'] == 0).sum(),
        "price_range": f"${df['Close'].min():.2f} - ${df['Close'].max():.2f}",
        "avg_volume": int(df['Volume'].mean()),
        "volatility": df['Close'].pct_change().std() * 100  # Annual volatility estimate
    }
    
    return metrics


def replay_stream(
    ticker: str,
    period: str = "2y",
    delay: float = 0.5,
    max_records: Optional[int] = None
) -> int:
    """
    Replay historical data with specified delay
    Useful for testing and demonstration
    
    Args:
        ticker: Stock ticker symbol
        period: Data period
        delay: Delay between records in seconds
        max_records: Maximum number of records to replay (None = all)
        
    Returns:
        Number of records replayed
    """
    try:
        df = fetch_historical_data(ticker, period)
        
        if max_records:
            df = df.head(max_records)
        
        logger.info(f"Replaying {len(df)} trading days for {ticker}")
        print(f"\nReplaying {len(df)} trading days for {ticker}\n")
        
        for idx, (timestamp, row) in enumerate(df.iterrows(), 1):
            data_point = {
                "ticker": ticker,
                "timestamp": timestamp.strftime('%Y-%m-%d'),
                "open": row['Open'],
                "high": row['High'],
                "low": row['Low'],
                "close": row['Close'],
                "volume": int(row['Volume'])
            }
            
            print(f"[{idx:4d}] {data_point['timestamp']} | "
                  f"O:{data_point['open']:7.2f} H:{data_point['high']:7.2f} "
                  f"L:{data_point['low']:7.2f} C:{data_point['close']:7.2f} "
                  f"V:{data_point['volume']:>10}")
            
            time.sleep(delay)
        
        logger.info(f"✓ Replayed {len(df)} records")
        return len(df)
    
    except Exception as e:
        logger.error(f"Error replaying data: {str(e)}")
        raise


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Fetch stock data")
    parser.add_argument("--ticker", default="AAPL", help="Stock ticker (default: AAPL)")
    parser.add_argument("--period", default="2y", help="Period (default: 2y)")
    parser.add_argument("--replay", action="store_true", help="Replay data with delays")
    parser.add_argument("--delay", type=float, default=0.5, help="Replay delay in seconds")
    parser.add_argument("--validate", action="store_true", help="Show data quality metrics")
    
    args = parser.parse_args()
    
    try:
        df = fetch_historical_data(args.ticker, args.period)
        
        if args.validate:
            metrics = validate_data_quality(df)
            print("\nData Quality Metrics:")
            print("=" * 50)
            for key, value in metrics.items():
                print(f"  {key}: {value}")
            print("=" * 50)
        
        if args.replay:
            replay_stream(args.ticker, args.period, args.delay)
        else:
            print(f"\n{df}")
    
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        exit(1)
