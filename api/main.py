from fastapi import FastAPI
from datetime import datetime
import json

app = FastAPI()

def parse_log():
    anomalies = []
    try:
        with open('logs/anomalies.log', 'r') as f:
            for line in f:
                parts = line.strip().split(' | ')
                if len(parts) >= 6:
                    anomalies.append({
                        "detected_at": parts[0],
                        "timestamp": parts[1],
                        "close": parts[2].split('=')[1],
                        "spread": parts[3].split('=')[1],
                        "z_score": parts[4].split('=')[1],
                        "score": parts[5].split('=')[1],
                        "change_pct": parts[6].split('=')[1] if len(parts) > 6 else None
                    })
    except FileNotFoundError:
        pass
    return anomalies

@app.get("/status")
def status():
    return {
        "status": "running",
        "time": str(datetime.now()),
        "service": "Stock Anomaly Detector"
    }

@app.get("/anomalies")
def get_anomalies():
    anomalies = parse_log()
    return {
        "total": len(anomalies),
        "anomalies": anomalies
    }

@app.get("/anomalies/count")
def get_count():
    anomalies = parse_log()
    return {"count": len(anomalies)}