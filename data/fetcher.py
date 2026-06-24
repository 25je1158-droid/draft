import yfinance as yf
import pandas as pd
import time

def fetch_historical_data(ticker: str, period: str = "max") -> pd.DataFrame:
    stock = yf.Ticker(ticker)
    df = stock.history(period=period)
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    df.dropna(inplace=True)
    return df

def replay_stream(ticker: str, delay: float = 0.5):
    df = fetch_historical_data(ticker)
    print(f"Replaying {len(df)} trading days for {ticker}")
    for timestamp, row in df.iterrows():
        data_point = {
            "ticker": ticker,
            "timestamp": str(timestamp),
            "open": row['Open'],
            "high": row['High'],
            "low": row['Low'],
            "close": row['Close'],
            "volume": row['Volume']
        }
        print(data_point)
        time.sleep(delay)

if __name__ == "__main__":
    replay_stream("AAPL")