import redis
import json
import time
from data.fetcher import fetch_historical_data

def produce_stream(ticker: str = "AAPL", stream_name: str = "stock_stream"):
    r = redis.Redis(host='localhost', port=6379, decode_responses=True)
    df = fetch_historical_data(ticker)
    print(f"Starting stream for {ticker} — {len(df)} data points")
    
    for timestamp, row in df.iterrows():
        data_point = {
            "ticker": ticker,
            "timestamp": str(timestamp),
            "open": str(row['Open']),
            "high": str(row['High']),
            "low": str(row['Low']),
            "close": str(row['Close']),
            "volume": str(row['Volume'])
        }
        r.xadd(stream_name, data_point)
        print(f"Produced: {data_point['timestamp']} close={data_point['close']}")
        time.sleep(0.3)

if __name__ == "__main__":
    produce_stream()