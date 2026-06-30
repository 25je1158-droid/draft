"""
Improved stream producer with error handling and configuration
"""
import redis
import json
import time
import logging
from typing import Optional

from config import Config
from data.fetcher import fetch_historical_data

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def connect_redis_with_retry(max_retries: int = 3, retry_delay: float = 1.0) -> redis.Redis:
    """
    Connect to Redis with exponential backoff retry logic
    
    Args:
        max_retries: Number of connection attempts
        retry_delay: Initial delay between retries in seconds
        
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
            logger.info(f"✓ Connected to Redis at {Config.REDIS_HOST}:{Config.REDIS_PORT}")
            return r
        
        except redis.ConnectionError as e:
            wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
            logger.warning(
                f"Redis connection attempt {attempt + 1}/{max_retries} failed: {str(e)}\n"
                f"Retrying in {wait_time}s..."
            )
            
            if attempt < max_retries - 1:
                time.sleep(wait_time)
            else:
                raise ConnectionError(
                    f"Failed to connect to Redis after {max_retries} attempts. "
                    f"Ensure Redis is running at {Config.REDIS_HOST}:{Config.REDIS_PORT}"
                )


def produce_stream(
    ticker: Optional[str] = None,
    stream_name: Optional[str] = None,
    period: Optional[str] = None,
    delay: Optional[float] = None
) -> int:
    """
    Stream historical stock data to Redis Streams
    
    Args:
        ticker: Stock ticker symbol (default: from config)
        stream_name: Redis stream name (default: from config)
        period: Data period (default: from config)
        delay: Delay between records in seconds (default: from config)
        
    Returns:
        Number of records streamed
    """
    # Use config defaults if not provided
    ticker = ticker or Config.STOCK_TICKER
    stream_name = stream_name or Config.REDIS_STREAM_NAME
    period = period or Config.DATA_PERIOD
    delay = delay if delay is not None else Config.STREAM_DELAY
    
    try:
        # Validate inputs
        if not ticker or len(ticker) < 1:
            raise ValueError("Invalid ticker symbol")
        if delay < 0:
            raise ValueError("Delay cannot be negative")
        
        # Connect to Redis
        r = connect_redis_with_retry()
        
        # Fetch historical data
        logger.info(f"Fetching historical data for {ticker}...")
        df = fetch_historical_data(ticker, period=period)
        
        if df.empty:
            logger.error(f"No data fetched for {ticker}")
            return 0
        
        logger.info(f"✓ Fetched {len(df)} data points for {ticker}")
        print(f"\nStarting stream for {ticker} — {len(df)} data points\n")
        
        # Stream data
        records_streamed = 0
        errors = 0
        
        try:
            for timestamp, row in df.iterrows():
                try:
                    # Prepare data point
                    data_point = {
                        "ticker": ticker,
                        "timestamp": timestamp.strftime('%Y-%m-%d'),
                        "open": str(row['Open']),
                        "high": str(row['High']),
                        "low": str(row['Low']),
                        "close": str(row['Close']),
                        "volume": str(int(row['Volume']))
                    }
                    
                    # Add to Redis stream
                    r.xadd(stream_name, data_point)
                    records_streamed += 1
                    
                    print(f"Produced: {data_point['timestamp']} close={data_point['close']}")
                    
                    # Delay before next record
                    time.sleep(delay)
                
                except Exception as e:
                    logger.error(f"Error streaming record {timestamp}: {str(e)}")
                    errors += 1
                    continue
        
        except KeyboardInterrupt:
            logger.info("Stream interrupted by user")
        except Exception as e:
            logger.error(f"Error during streaming: {str(e)}")
            raise
        
        logger.info(f"✓ Streamed {records_streamed} records with {errors} errors")
        print(f"\n{'='*60}")
        print(f"Stream completed: {records_streamed} records streamed")
        print(f"Stream name: {stream_name}")
        print(f"Errors: {errors}")
        print(f"{'='*60}\n")
        
        return records_streamed
    
    except Exception as e:
        logger.error(f"Fatal error in produce_stream: {str(e)}")
        raise


def clear_stream(stream_name: Optional[str] = None) -> bool:
    """
    Clear all messages from a Redis stream (useful for testing)
    
    Args:
        stream_name: Redis stream name (default: from config)
        
    Returns:
        True if successful
    """
    stream_name = stream_name or Config.REDIS_STREAM_NAME
    
    try:
        r = connect_redis_with_retry()
        r.delete(stream_name)
        logger.info(f"✓ Cleared stream: {stream_name}")
        return True
    except Exception as e:
        logger.error(f"Error clearing stream: {str(e)}")
        return False


def get_stream_info(stream_name: Optional[str] = None) -> dict:
    """
    Get information about a Redis stream
    
    Args:
        stream_name: Redis stream name (default: from config)
        
    Returns:
        Dictionary with stream info
    """
    stream_name = stream_name or Config.REDIS_STREAM_NAME
    
    try:
        r = connect_redis_with_retry()
        info = r.xinfo_stream(stream_name)
        logger.info(f"Stream info for {stream_name}: {info}")
        return info
    except redis.ResponseError:
        logger.warning(f"Stream {stream_name} does not exist")
        return {"length": 0}
    except Exception as e:
        logger.error(f"Error getting stream info: {str(e)}")
        return {}


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Stream stock data to Redis")
    parser.add_argument("--ticker", default=Config.STOCK_TICKER, help="Stock ticker (default: AAPL)")
    parser.add_argument("--stream", default=Config.REDIS_STREAM_NAME, help="Stream name (default: stock_stream)")
    parser.add_argument("--period", default=Config.DATA_PERIOD, help="Data period (default: 2y)")
    parser.add_argument("--delay", type=float, default=Config.STREAM_DELAY, help="Delay between records (default: 0.3s)")
    parser.add_argument("--clear", action="store_true", help="Clear stream before starting")
    
    args = parser.parse_args()
    
    try:
        if args.clear:
            logger.info(f"Clearing stream: {args.stream}")
            clear_stream(args.stream)
        
        produce_stream(
            ticker=args.ticker,
            stream_name=args.stream,
            period=args.period,
            delay=args.delay
        )
    
    except Exception as e:
        logger.error(f"Stream producer failed: {str(e)}")
        exit(1)
