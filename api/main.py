"""
Improved API with error handling, validation, and better data management
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks
from datetime import datetime
import json
import logging
from pathlib import Path
from typing import List, Optional
from pydantic import BaseModel

from config import Config

# Setup logging
Config.ensure_log_dir()
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(Config.LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Stock Anomaly Detector API",
    description="Real-time anomaly detection for stock market data",
    version="1.0.0"
)


# Pydantic models for validation
class AnomalyRecord(BaseModel):
    """Schema for anomaly records"""
    detected_at: str
    timestamp: str
    close: float
    spread: float
    z_score: float
    score: float
    change_pct: float

    class Config:
        json_schema_extra = {
            "example": {
                "detected_at": "2026-06-29 15:35:24",
                "timestamp": "2023-03-14",
                "close": 148.22,
                "spread": 6.42,
                "z_score": -2.85,
                "score": -0.184,
                "change_pct": -0.051
            }
        }


class AnomaliesResponse(BaseModel):
    """Schema for anomalies list response"""
    total: int
    anomalies: List[AnomalyRecord]


class StatusResponse(BaseModel):
    """Schema for status response"""
    status: str
    service: str
    time: str
    config: Optional[dict] = None


def parse_log_safely() -> List[dict]:
    """
    Safely parse anomalies log file with error handling
    Returns list of anomaly records
    """
    anomalies = []
    
    if not Path(Config.LOG_FILE).exists():
        logger.warning(f"Log file not found: {Config.LOG_FILE}")
        return anomalies
    
    try:
        with open(Config.LOG_FILE, 'r') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    line = line.strip()
                    if not line:
                        continue
                    
                    parts = line.split(' | ')
                    
                    if len(parts) < 7:
                        logger.warning(f"Line {line_num}: Invalid format (expected 7 fields, got {len(parts)})")
                        continue
                    
                    # Parse each field with validation
                    try:
                        detected_at = parts[0]
                        timestamp = parts[1]
                        close = float(parts[2].split('=')[1])
                        spread = float(parts[3].split('=')[1])
                        z_score = float(parts[4].split('=')[1])
                        score = float(parts[5].split('=')[1])
                        change_pct = float(parts[6].split('=')[1])
                        
                        anomalies.append({
                            "detected_at": detected_at,
                            "timestamp": timestamp,
                            "close": close,
                            "spread": spread,
                            "z_score": z_score,
                            "score": score,
                            "change_pct": change_pct
                        })
                    except (ValueError, IndexError) as e:
                        logger.warning(f"Line {line_num}: Failed to parse values - {str(e)}")
                        continue
                        
                except Exception as e:
                    logger.error(f"Line {line_num}: Unexpected error - {str(e)}")
                    continue
                    
    except IOError as e:
        logger.error(f"Failed to read log file: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to read anomaly log")
    except Exception as e:
        logger.error(f"Unexpected error reading log: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")
    
    return anomalies


@app.get("/", tags=["Health"])
def root():
    """Root endpoint - API alive check"""
    return {
        "message": "Stock Anomaly Detector API",
        "docs": "/docs",
        "version": "1.0.0"
    }


@app.get("/status", response_model=StatusResponse, tags=["Status"])
def status():
    """
    Get service status and configuration
    """
    try:
        return {
            "status": "running",
            "service": "Stock Anomaly Detector",
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "config": {
                "ticker": Config.STOCK_TICKER,
                "contamination": Config.ISOLATION_FOREST_CONTAMINATION,
                "rolling_window": Config.ROLLING_WINDOW
            }
        }
    except Exception as e:
        logger.error(f"Status endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get status")


@app.get("/health", tags=["Health"])
def health_check():
    """
    Lightweight health check endpoint
    """
    return {"status": "ok", "timestamp": datetime.now().isoformat()}


@app.get("/anomalies", response_model=AnomaliesResponse, tags=["Anomalies"])
def get_anomalies(limit: Optional[int] = None, offset: Optional[int] = 0):
    """
    Get all detected anomalies
    
    Query Parameters:
    - limit: Maximum number of anomalies to return (default: all)
    - offset: Number of anomalies to skip (default: 0)
    """
    try:
        anomalies = parse_log_safely()
        
        # Validate pagination parameters
        if offset and offset < 0:
            raise HTTPException(status_code=400, detail="offset must be non-negative")
        if limit and limit < 1:
            raise HTTPException(status_code=400, detail="limit must be positive")
        
        # Apply pagination
        offset = offset or 0
        if limit:
            paginated = anomalies[offset:offset + limit]
        else:
            paginated = anomalies[offset:]
        
        return {
            "total": len(anomalies),
            "anomalies": paginated
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching anomalies: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to fetch anomalies")


@app.get("/anomalies/count", tags=["Anomalies"])
def get_count():
    """
    Get the count of detected anomalies
    """
    try:
        anomalies = parse_log_safely()
        return {
            "count": len(anomalies),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting anomaly count: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get anomaly count")


@app.get("/anomalies/latest", response_model=Optional[AnomalyRecord], tags=["Anomalies"])
def get_latest_anomaly():
    """
    Get the most recently detected anomaly
    """
    try:
        anomalies = parse_log_safely()
        if not anomalies:
            return None
        return anomalies[-1]
    except Exception as e:
        logger.error(f"Error fetching latest anomaly: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to fetch latest anomaly")


@app.get("/anomalies/range", response_model=AnomaliesResponse, tags=["Anomalies"])
def get_anomalies_by_date_range(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
):
    """
    Get anomalies within a date range
    
    Query Parameters:
    - start_date: Start date (YYYY-MM-DD format)
    - end_date: End date (YYYY-MM-DD format)
    """
    try:
        anomalies = parse_log_safely()
        
        if not start_date and not end_date:
            return {"total": len(anomalies), "anomalies": anomalies}
        
        filtered = []
        for anomaly in anomalies:
            timestamp = anomaly["timestamp"]
            if start_date and timestamp < start_date:
                continue
            if end_date and timestamp > end_date:
                continue
            filtered.append(anomaly)
        
        return {
            "total": len(filtered),
            "anomalies": filtered
        }
    except Exception as e:
        logger.error(f"Error filtering anomalies by date: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to filter anomalies")


@app.get("/anomalies/stats", tags=["Analytics"])
def get_anomaly_statistics():
    """
    Get statistical summary of detected anomalies
    """
    try:
        anomalies = parse_log_safely()
        
        if not anomalies:
            return {
                "total_count": 0,
                "message": "No anomalies detected"
            }
        
        import statistics
        
        closes = [a["close"] for a in anomalies]
        spreads = [a["spread"] for a in anomalies]
        scores = [a["score"] for a in anomalies]
        
        return {
            "total_count": len(anomalies),
            "close_price": {
                "mean": statistics.mean(closes),
                "median": statistics.median(closes),
                "stdev": statistics.stdev(closes) if len(closes) > 1 else 0,
                "min": min(closes),
                "max": max(closes)
            },
            "daily_spread": {
                "mean": statistics.mean(spreads),
                "median": statistics.median(spreads),
                "max": max(spreads)
            },
            "anomaly_scores": {
                "mean": statistics.mean(scores),
                "min": min(scores),
                "max": max(scores)
            }
        }
    except Exception as e:
        logger.error(f"Error calculating statistics: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to calculate statistics")


@app.exception_handler(404)
async def not_found_handler(request, exc):
    """Custom 404 handler"""
    return {
        "error": "Not Found",
        "message": "The requested endpoint does not exist",
        "path": request.url.path
    }


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    """Custom 500 handler"""
    logger.error(f"Internal server error: {str(exc)}")
    return {
        "error": "Internal Server Error",
        "message": "An unexpected error occurred"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=Config.API_HOST,
        port=Config.API_PORT,
        reload=Config.API_RELOAD
    )
