# Stock Market Anomaly Detection System
A real-time stock market anomaly detection system that streams historical stock data, extracts statistical features, detects abnormal trading behavior using Isolation Forest, logs anomalies, and exposes results through a FastAPI REST API.
The project simulates a live market data pipeline using Redis Streams while performing anomaly detection on historical stock prices fetched from Yahoo Finance.
## What it does
The system fetches historical stock market data,simulates a real-time data stream,computes statistical indicators,detects anomalous trading days using machine learning algorithms,stores the detected anomalies in log files and displays them using REST API.
### Workflow
1. Fetch historical stock data from Yahoo Finance.
2. Stream data using redis.
3. Compute rolling statistical parameters.
4. Train an isolation forest model.
5. Detect anomalous trading activity.
6. Store anomalies in log files.
7. Access results using API.
## System Architecture
Data flows from yfinance → Redis Stream → Anomaly Detector → Log File → FastAPI
## Components
### 1.Data Fetcher
- Downloads data using Yahoo Finance(yfinance).
- Retrieves
  - Open
  - High
  - Low
  - Close
  - Volume
- removes Missing values
### 2.Stream Producer
Simulates real-time data by:
- Reading historical prices
- Storing each day to redis stream
- Add configurable delay between records
- Produce market events
### 3.Feature Engineering
Some new features are engineered to train isolation forest model
- 20-day rolling mean
- 20-day rolling standard deviation
- Z-score
- Daily Percentage Price Change
- Trading Volume
### 4.Anomaly Detection System
The system uses Isolation Forest from Scikit-learn.
#### Characteristics:
- Algorithm: Isolation Forest
- Contamination: 3%
- Random State: 42
#### Generated Outputs:
- Anomaly Score
Detected anomalies are stored to:
logs/anomalies.log
Each log entry contains:
- Detection timestamp
- Trading date
- Closing price
- Daily spread
- Z-score
- Isolation Forest score
- Daily percentage change
### 5. REST API
Built using FastAPI
#### GET /status
Returns service status and current server time.
Example response:
{
  "status": "running",
  "service": "Stock Anomaly Detector",
  "time": "2026-06-11 15:40:02"
}
#### GET /anomalies
Returns all detected anomalies.
Example response:
{
  "total": 15,
  "anomalies": [
    {
      "timestamp": "2023-03-14",
      "close": "148.22",
      "spread": "6.42",
      "z_score": "-2.85",
      "score": "-0.184",
      "change_pct": "-0.051"
    }
  ]
}
#### GET /anomalies/count
Returns only the number of detected anomalies.
Example response:
{
  "count": 15
}
## How to Run
1. Install dependencies: pip install -r requirements.txt
2. Start Memurai/Redis server
3. Run anomaly detector: python -m detector.anomaly
4. Start API server: python -m uvicorn api.main:app --reload
5. Access API at http://localhost:8000
## Sample Output
Starting stream for AAPL — 503 data points

Produced: 2024-01-02 close=185.32

Produced: 2024-01-03 close=184.70

...

ANOMALY DETECTED:
- 2026-06-11 15:35:24
- close=173.44
- spread=8.25
- z_score=-2.91
- score=-0.1821
- change_pct=-0.0542

Total anomalies detected: 16 out of 484 trading days
## Tech Stack
- Language: Python
- API Framework: FastAPI
- Data Source: yfinance
- Data Processing: Numpy,Pandas
- Machine Learning: Isolation Forest(Scikit-learn)
- Streaming: Redis
- Server: Uvicorn
- Logging: Python I/O
## Key Design Decisions
- **Isolation Forest** was selected because it efficiently detects anomalies in high-dimensional financial datasets without requiring labeled examples. Contamination to isolated forest set to 3% since 1-2% is too less and actual outliers might be missed upon and anything more than 4-5% is too high and contains noise.Also 3% would approximately coresspond to 3 significant market events per year which match historical reality for AAPL.
- **Isolation Forest** was chosen over **Z-Score** because simple z score would flag a week of continuous high trades because it's a single feature as outlier whereas isolation forest works with numerous features at a time and entries not clearing maximum checkpoints are declared outliers hence it is more useful.
- **Redis Streams** simulate a real-time stock data feed while maintaining a decoupled architecture between producers and consumers.
- **Redis stream** was chosen over queue because a regular queue is destructive — once a message is consumed it's gone. If the detector crashes mid-processing, the data is lost permanently. Redis Streams are persistent and replayable — every message stays in the stream with a unique ID. If the detector crashes, it can resume from exactly where it left off. Multiple consumers can also read the same stream independently.
- Rolling statistical features (rolling mean, standard deviation, z-score, and price changes) capture local market behavior and improve anomaly detection accuracy. 
- **Idea behind selection of features** Daily_Spread (catures variance of the day (high- low) unusually high volatality showcases abnormality), Z_Score(measure of how close daily data is from rolling mean of 20 days), Price_Change_Percentage(captures single day deviation used when there exists a series of consecutive highs or lows since otherwise the entirety of days showcasing abnormal high or low would get flagged) and Volume(a confirmation signal anomalous price on high volume shows real market activity rather than thin market noise)
- To handle Redis failure, the producer implements retry with exponential backoff, and Redis AOF persistence can be enabled to recover stream state on restart.