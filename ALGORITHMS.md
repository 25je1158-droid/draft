# Algorithms & Data Structures: Stock Anomaly Detector

## 1. Core Algorithm: Isolation Forest

### 1.1 What is Isolation Forest?

Isolation Forest is an **unsupervised anomaly detection algorithm** that isolates outliers by randomly selecting features and split values, then measuring how many splits are needed to isolate each point.

**Key Insight**: Anomalies are easier to isolate than normal points.

```
Normal data distribution:
    ●●●●●
    ●●●●●    Many splits needed to isolate each point
    ●●●●●

Anomalous data (outlier):
    ●●●●●
    ●●●●●    ★ Only 2-3 splits needed to isolate
    ●●●●★


Isolation Forest creates random trees and measures:
  - Path length to isolate a point
  - Anomalies have shorter paths (isolated faster)
  - Anomaly score = average path length across trees
```

### 1.2 How It Works (Step-by-Step)

```python
# Input: 4D data for each trading day
#   [daily_spread, z_score, Volume, price_change_pct]

# STEP 1: Build Isolation Trees (100 trees)
for tree_i in range(100):
    # Random sample of data (256 points)
    sample = random_sample(data, size=256)
    
    # Build tree by random splits
    tree = build_random_tree(sample)
    # At each node: pick random feature + random split value
    # Continue until: each point isolated OR max depth reached

# STEP 2: Calculate Anomaly Score
for each_point in data:
    path_lengths = []
    for each_tree in 100_trees:
        path = traverse(each_tree, point)  # How many splits to reach leaf?
        path_lengths.append(len(path))
    
    # Average path length across all trees
    avg_path = mean(path_lengths)
    
    # Normalize to anomaly score
    anomaly_score = 2^(-avg_path / c)  # c = avg path of random tree
    # Score in [0, 1]: 1 = anomalous, 0 = normal

# STEP 3: Threshold
anomalies = points where anomaly_score > threshold
```

### 1.3 Complexity Analysis

```
Training (Batch):
  Time:  O(t * n log n)  where t = trees (100), n = records (500)
         = O(100 * 500 * log(500))
         = O(100 * 500 * 9)
         ≈ 450,000 operations
         ≈ 50ms on modern CPU
  
  Space: O(t * n) for storing all trees
         = O(100 * 500) 
         = 50,000 nodes
         ≈ 2-5 MB

Prediction (Per Record):
  Time:  O(t * depth)  where depth ≈ log(n)
         = O(100 * log(500))
         = O(100 * 9)
         ≈ 0.1ms per record
  
  Space: O(1) (just traverse path)
```

### 1.4 Why Isolation Forest for Stock Anomalies?

| Aspect | Isolation Forest | Alternative | Winner |
|--------|------------------|-------------|--------|
| **Unsupervised** | ✅ No labels needed | ❌ Need anomaly examples | ✅ IF |
| **Multi-dimensional** | ✅ Handles correlations | ❌ z-score ignores them | ✅ IF |
| **Time Complexity** | O(n log n) | O(n²) for clustering | ✅ IF |
| **Interpretability** | ⚠️ Black box | ✅ Clear rules | ❌ IF |
| **Scalability** | ✅ Handles 1M+ | ⚠️ Expensive | ✅ IF |

**Real-World Example**:
```
Day 1: Close=$160, Volume=2M, Spread=$5, Change=+1%
  → Normal: Low z-score (-0.5), low volume, normal spread
  → IF score: 0.1 (normal)

Day 2: Close=$200, Volume=50M, Spread=$30, Change=+25%
  → Anomalous: High z-score (3.5), extreme volume, huge spread
  → IF score: 0.95 (highly anomalous)
  
Day 3: Close=$158, Volume=100k, Spread=$2, Change=+0.1%
  → Normal: Low everything
  → IF score: 0.05 (very normal)
```

---

## 2. Feature Engineering

### 2.1 Features Used

#### **Feature 1: Daily Spread**
```
Formula: spread = High - Low

Why:
  - Captures intraday volatility
  - High spread = uncertain trading (anomalous)
  
Example:
  Normal day:   High=$170, Low=$160 → Spread=$10
  Volatile day: High=$190, Low=$140 → Spread=$50 ← Anomaly signal
  
Time Complexity: O(n) - vectorized subtraction
Space Complexity: O(n) - new column
```

#### **Feature 2: Rolling Mean (20-day)**
```
Formula: rolling_mean = Close.rolling(window=20).mean()

Why:
  - Smooths short-term noise
  - Captures long-term trend
  - Used as baseline for z-score
  
Example:
  Day 1-19:   rolling_mean = NaN (not enough history)
  Day 20:     rolling_mean = avg(Close[0:20])
  Day 21:     rolling_mean = avg(Close[1:21])  ← Sliding window
  
Time Complexity: O(n * w) = O(n * 20) = O(n)
Space Complexity: O(n)
```

#### **Feature 3: Rolling Standard Deviation**
```
Formula: rolling_std = Close.rolling(window=20).std()

Why:
  - Measures volatility over time
  - High std = high uncertainty
  - Used to normalize z-score
  
Example:
  Normal market: std = $2-3
  Volatile market: std = $5-10
  
Time Complexity: O(n * w) = O(n)
Space Complexity: O(n)
```

#### **Feature 4: Z-Score (Normalized Deviation)**
```
Formula: z_score = (Close - rolling_mean) / rolling_std

Why:
  - Standardizes price deviation
  - Values > 3 are statistically unusual
  - Comparable across different stocks
  
Example:
  Normal day:    z_score = 0.5    ← Within 1 std
  Unusual day:   z_score = 2.5    ← 2.5 stds away
  Extreme day:   z_score = -4.5   ← 4.5 stds away (anomalous!)
  
Interpretation:
  |z| < 1:   68% of data (normal)
  |z| < 2:   95% of data (normal)
  |z| < 3:   99.7% of data (normal)
  |z| > 3:   < 0.3% of data (anomalous!)
  
Time Complexity: O(n)
Space Complexity: O(n)
```

#### **Feature 5: Price Change Percentage**
```
Formula: price_change_pct = Close.pct_change()
        = (Close[i] - Close[i-1]) / Close[i-1]

Why:
  - Captures daily momentum
  - Large changes are anomalous signals
  - Useful for detecting flash crashes/rallies
  
Example:
  Normal change:  +1% (day-to-day natural move)
  Large change:   +10% (unusual, possible anomaly)
  Crash:          -15% (very anomalous!)
  
Time Complexity: O(n)
Space Complexity: O(n)
```

### 2.2 Feature Engineering Pipeline

```
Input Data (500 records):
┌─────────────────────────────────────────┐
│ Date    | Open   | High   | Low   | Close | Volume
│ 2024-07-01 | 150 | 160 | 140 | 155 | 2M
│ 2024-07-02 | 155 | 165 | 150 | 160 | 2.1M
│ ...     | ...    | ...    | ...   | ...   | ...
└─────────────────────────────────────────┘
         ↓ Step 1: Calculate spreads
┌─────────────────────────────────────────┐
│ daily_spread = High - Low
│ 2024-07-01: $20
│ 2024-07-02: $15
└─────────────────────────────────────────┘
         ↓ Step 2: Calculate rolling stats
┌─────────────────────────────────────────┐
│ rolling_mean = Close.rolling(20).mean()
│ rolling_std = Close.rolling(20).std()
│ First 19 rows: NaN (not enough data)
└─────────────────────────────────────────┘
         ↓ Step 3: Calculate z-scores
┌─────────────────────────────────────────┐
│ z_score = (Close - rolling_mean) / rolling_std
│ 2024-07-01: NaN
│ ...
│ 2024-07-20: 0.45
│ 2024-07-21: -1.23
└─────────────────────────────────────────┘
         ↓ Step 4: Calculate price changes
┌─────────────────────────────────────────┐
│ price_change_pct = Close.pct_change()
│ 2024-07-01: NaN (no previous day)
│ 2024-07-02: +3.2%
└─────────────────────────────────────────┘
         ↓ Step 5: Drop NaN rows
┌─────────────────────────────────────────┐
│ Final dataset: 481 records (19 dropped)
│ Columns: [Open, High, Low, Close, Volume,
│           daily_spread, rolling_mean,
│           rolling_std, z_score,
│           price_change_pct]
└─────────────────────────────────────────┘
         ↓ Step 6: Feature scaling (optional)
┌─────────────────────────────────────────┐
│ X = [daily_spread, z_score, Volume, price_change_pct]
│ Ready for model training
└─────────────────────────────────────────┘
```

### 2.3 Why These Features?

```
Question: Why not just use Close price?

Answer: Single feature is insufficient

Example:
  Day A: Close=$200  ← 30% increase
    → Single feature: Could be normal or anomalous
    → Multi-feature: z_score=3.5, volume=normal, spread=$10
       → Verdict: Anomalous (unexpected jump)
  
  Day B: Close=$200  ← 30% increase (same as Day A)
    → Single feature: Could be normal or anomalous
    → Multi-feature: z_score=0.5 (expected), volume=50M, spread=$40
       → Verdict: Normal (planned earnings announcement)

Multi-dimensional features catch context!
```

---

## 3. Data Structures Used

### 3.1 DataFrame (Pandas)

```python
# Core data structure: Pandas DataFrame

Why DataFrame?
  ✅ Vectorized operations (100x faster than loops)
  ✅ Built-in time series support (dates as index)
  ✅ Missing data handling (NaN, fillna, dropna)
  ✅ Column-wise filtering (easy feature selection)
  ✅ SQL-like operations (groupby, merge, rolling)

Example:
df = pd.DataFrame({
    'Open': [150, 155, 160, ...],      # 500 rows
    'High': [160, 165, 170, ...],
    'Low': [140, 145, 150, ...],
    'Close': [155, 160, 165, ...],
    'Volume': [2M, 2.1M, 2.2M, ...]
}, index=pd.date_range('2024-07-01', periods=500))

# Time Complexity:
#   Single row access:        O(1)
#   Column access:            O(1)
#   Row filtering:            O(n)
#   Rolling operations:       O(n*w)
#   Sorting:                  O(n log n)

# Space Complexity:
#   Storage:                  O(n * m) where m=5 columns
#   = O(500 * 5) = 2500 values ≈ 40KB
```

### 3.2 IsolationForest Model (Scikit-learn)

```python
# Trained model stored as object

model = IsolationForest(
    contamination=0.03,    # Expect 3% anomalies
    random_state=42,       # Reproducible
    n_estimators=100       # 100 random trees
)
model.fit(X)              # Train

# Internal structure:
# - 100 decision trees
# - Each tree has splits on random features
# - Path length determines anomaly score

# Time Complexity:
#   model.predict(X):         O(t * log n)  where t=100 trees
#   model.decision_function(): O(t * log n)

# Space Complexity:
#   Model size:                O(100 * log n)
#   ≈ 2-5 MB for 500 records
```

### 3.3 Redis Stream (Optional - Streaming)

```python
# Data structure: Redis Streams (like Kafka topics)

Purpose: Queue of stock data records

Entry format:
{
  'timestamp': '2026-07-02T10:00:00',
  'close': '155.5',
  'volume': '2000000',
  'spread': '10.2'
}

Operations:
  XADD stream_name * field value  → Add entry (O(1))
  XREAD stream_name               → Read entries (O(n))
  XRANGE stream_name start end    → Range query (O(m))

Consumer groups:
  ├─ Multiple consumers read same stream
  ├─ At-least-once semantics (no data loss)
  ├─ Replay capability (read from start)
```

---

## 4. Time Complexity Cheat Sheet

```
Operation                          | Complexity | Example (500 records)
-----------------------------------+------------+----------------------
Fetch from Yahoo Finance           | O(1) API   | ~500ms network
DataFrame creation                 | O(n)       | ~5ms
Calculate daily_spread (High-Low)  | O(n)       | ~1ms
Calculate rolling_mean             | O(n*w)     | O(n*20) = ~5ms
Calculate rolling_std              | O(n*w)     | O(n*20) = ~5ms
Calculate z_score                  | O(n)       | ~2ms
Calculate price_change_pct         | O(n)       | ~2ms
Drop NaN rows                      | O(n)       | ~1ms
---
Isolation Forest training          | O(n log n) | ~50ms
Prediction (1 record)              | O(log n)   | ~0.1ms
Prediction (all 500 records)       | O(n log n) | ~50ms
---
API response (with DB)             | O(1)       | ~5ms
---
TOTAL PIPELINE TIME                |            | ~580ms ✅
```

---

## 5. Space Complexity Analysis

```
Component                  | Space | Notes
----------------------------+-------+----------------------------------------
Raw OHLCV data (500 rows)  | 40KB  | 5 columns × 8 bytes × 500 rows
Engineered features        | 80KB  | 10 columns total
Isolation Forest model     | 5MB   | 100 trees × ~50KB each
Anomalies log file         | 10KB  | ~15 anomalies × 500 bytes
Redis Streams (if used)    | 50KB  | 500 entries × 100 bytes each
---
TOTAL                      | ~6MB  | Minimal memory footprint
```

---

## 6. Algorithm Comparison Table

```
Algorithm              | Time Complexity | Space | Interpretability | Supervised?
-----------------------+-----------------+-------+------------------+------------
Z-Score Threshold      | O(n)            | O(n)  | ✅ Clear rules   | ❌ No
K-Means Clustering     | O(n²) iterative | O(n)  | ⚠️ Medium        | ❌ No
Isolation Forest       | O(n log n)      | O(n)  | ❌ Black box     | ❌ No
DBSCAN                 | O(n²)           | O(n)  | ⚠️ Medium        | ❌ No
Neural Network (LSTM)  | O(n * layers)   | O(n)  | ❌ Black box     | ✅ Yes (need labels)
Random Forest (labeled)| O(n log n)      | O(n)  | ⚠️ Medium        | ✅ Yes

★ Our Choice: Isolation Forest ★
```

---

## 7. Edge Cases & Handling

### 7.1 Empty Data

```python
# Input: Empty DataFrame
df = pd.DataFrame()

# Processing:
features_df = calculate_features(df)  # Returns empty DataFrame
model = train_detector(features_df)   # Raises ValueError

# Error:
"Insufficient data for training (need at least 10 rows)"

# Time Complexity: O(1) (immediate check)
```

### 7.2 Constant Prices

```python
# Input: Close price never changes
close = [160, 160, 160, 160, ...]

# Processing:
rolling_std = 0  (no variation)
z_score = (160 - 160) / 0  = NaN ← Division by zero!
dropna() removes ALL rows

# Handling:
detected in test_constant_prices()
raises ValueError: "Insufficient data for training"

# Time Complexity: O(n) for detection
```

### 7.3 Single Row

```python
# Input: Only 1 trading day
df = pd.DataFrame([AAPL_2026_07_02])

# Processing:
rolling_mean = NaN (need 20 days)
rolling_std = NaN
z_score = NaN
dropna() removes the 1 row

# Result: Empty DataFrame
# Time Complexity: O(1) for detection
```

### 7.4 Missing Data (NaN)

```python
# Input: Some missing values
Close = [160, 161, NaN, 163, 164, ...]

# Processing:
dropna() removes NaN rows
Preserved 95% of data (acceptable)

# Example:
Original: 500 rows
After dropna: 481 rows (dropped 19 from rolling window)
Expected: Some more dropped from input NaN
Result: ~470 rows

# Time Complexity: O(n) for dropna
```

---

## 8. Scaling Considerations

### 8.1 Current Approach (Works well)

```
Records: 500
Features: 10
Computation: Sequential
Time: ~600ms
Space: ~6MB
Bottleneck: Network (fetch data)
```

### 8.2 Scale to 50K Records

```
Challenge: Memory & CPU time

Solution 1: Batch Processing
  ├─ Process 1000 records at a time
  ├─ Combine predictions
  └─ Time: ~1s per batch

Solution 2: Optimize Pandas
  ├─ Use dtype optimization (float32 instead of float64)
  ├─ Use chunked reading
  └─ Memory: 300MB → 150MB

Solution 3: Caching
  ├─ Cache rolling statistics
  ├─ Avoid recalculation
  └─ Time: -50% from previous run

Time: ~10-20s (from ~600ms)
Space: ~300MB
```

### 8.3 Scale to 1M Records

```
Challenge: CPU-bound, not I/O-bound anymore

Solution: Distributed Computing (Apache Spark)

Architecture:
  Raw Data → Kafka → Spark Cluster → Model Server
  
  Spark benefits:
    ├─ Distributed DataFrame processing
    ├─ Lazy evaluation (efficient)
    ├─ MLlib for distributed Isolation Forest
    └─ Handles out-of-core data

Time: ~30s for 1M records (parallelized)
Space: Distributed (each node ~1GB)
```

### 8.4 Scaling Timeline

```
Scale     | Current | Batch  | Distributed | Tech
----------+---------+--------+-------------+------------------
500       | 0.6s    | —      | —           | Pandas ✅
5K        | 5s      | 1s     | —           | Pandas (optimized)
50K       | 50s     | 10s    | —           | Pandas (batched)
500K      | 500s    | 100s   | —           | Pandas (painful)
5M        | 5000s   | 1000s  | 30s ✅      | Spark + Distributed
50M       | —       | —      | 300s        | Spark + GPU
```

---

## 9. Algorithm Interview Walkthrough

**Q: "Explain Isolation Forest to someone who hasn't heard of it"**

A: "Isolation Forest detects anomalies by isolating them. Think of it like finding counterfeits in a batch of coins:
   - Counterfeit coins have weird weight (anomaly)
   - Normal coins cluster together (majority)
   - Isolation Forest 'splits' the data randomly on features
   - Anomalies get isolated faster (fewer splits needed)
   - We measure path length: shorter path = more anomalous
   
   Time: O(n log n) training, O(log n) per prediction
   Space: O(n) for storing trees
   Best for: Unsupervised, multi-dimensional data"

**Q: "Why not just use standard deviation (z-score)?"**

A: "Z-score works in 1D but misses patterns:
   - Example: Day has high volume AND price spike
   - Z-score sees: High volume (normal) + price spike (anomalous)
   - Result: Conflicting signals
   
   Isolation Forest sees both together = strong anomaly signal
   It captures correlations that single features can't"

**Q: "What if your model breaks? What's your fallback?"**

A: "Three layers of fallback:
   1. Model fails to train → Fall back to z-score threshold
   2. API goes down → Return cached results
   3. All systems fail → Return HTTP 503 (service unavailable)
   
   Tests ensure we catch these before production"

---

## 10. Key Takeaways

```
1. Algorithm:        Isolation Forest
   ├─ Why: Unsupervised, O(n log n), handles multi-dim
   └─ Alternative: Could use z-score (simpler) or LSTM (needs labels)

2. Features:         5 engineered features
   ├─ Spread, Rolling Stats, Z-Score, Price Change %
   └─ Why: Captures volatility, momentum, deviation from norm

3. Data Structure:   Pandas DataFrame
   ├─ Time: Vectorized O(n) operations
   └─ Space: Minimal (6MB for 500 records)

4. Complexity:       O(n log n) training, O(log n) per prediction
   ├─ Scalable to 100K records on laptop
   └─ Needs Spark for 1M+ records

5. Production Ready: Yes
   ├─ Error handling for edge cases
   ├─ Comprehensive testing (21 tests)
   └─ Monitoring hooks for degradation
```

---

**Version**: 1.0  
**Last Updated**: July 2, 2026  
**For Interview**: Read sections 1.2, 2.1-2.2, 4, and 9 for Q&A prep
