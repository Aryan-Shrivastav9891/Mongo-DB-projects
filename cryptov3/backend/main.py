from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
# Try to import main predictor, fall back to alternative if TensorFlow not available
try:
    from services.prediction_service import CandlestickPredictor
    print("Using main prediction service")
except ImportError:
    from services.prediction_service_alt import CandlestickPredictor
    print("Using alternative prediction service (no TensorFlow)")
import shutil
import os
import hashlib
import numpy as np
from tempfile import NamedTemporaryFile
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
import random
from datetime import datetime

app = FastAPI(
    title="Candlestick Chart Prediction API",
    description="API for predicting candlestick chart patterns and trends",
    version="1.0.0"
)

# Add CORS middleware to allow cross-origin requests from frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-Requested-With"],
    expose_headers=["Content-Length", "Content-Disposition"]
)

# Initialize the predictor
predictor = CandlestickPredictor()

# Create a prediction cache to ensure consistent results for the same image
prediction_cache = {}

@app.get("/")
def read_root():
    return {"status": "API is running", "message": "Welcome to Candlestick Chart Prediction API"}

@app.get("/api/health")
def health_check():
    """Health check endpoint to verify API is functioning"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "cache_size": len(prediction_cache)
    }

@app.post("/api/predict")
async def predict_chart(
    image: UploadFile = File(...),
    timeframe: str = Form(...)
):
    print(f"Received predict request with timeframe: {timeframe}")
    print(f"Image filename: {image.filename}, content type: {image.content_type}")
    
    # Validate timeframe
    valid_timeframes = ["15m", "1h", "4h", "1d", "7d"]
    if timeframe not in valid_timeframes:
        print(f"Invalid timeframe: {timeframe}")
        return {"error": f"Invalid timeframe. Must be one of: {', '.join(valid_timeframes)}"}
    
    # Save uploaded file temporarily
    try:
        print("Reading image content...")
        # Read the image content for hashing
        image_content = await image.read()
        print(f"Read {len(image_content)} bytes of image data")
        
        # Generate a hash of the image content to use as cache key
        image_hash = hashlib.md5(image_content).hexdigest()
        cache_key = f"{image_hash}_{timeframe}"
        print(f"Generated image hash: {image_hash}")
        
        # If this image with this timeframe has been processed before, return cached result
        if cache_key in prediction_cache:
            print(f"Using cached result for image: {image_hash}")
            return prediction_cache[cache_key]
        
        # Reset file position for reading again
        print("Resetting file position...")
        await image.seek(0)
        
        # Save to temporary file
        print("Saving to temporary file...")
        try:
            # Create a directory for temp files if it doesn't exist
            temp_dir = os.path.join(os.path.dirname(__file__), "temp_uploads")
            os.makedirs(temp_dir, exist_ok=True)
            
            # Use a more reliable method to save the file
            temp_path = os.path.join(temp_dir, f"temp_{image_hash}{os.path.splitext(image.filename)[1]}")
            with open(temp_path, "wb") as temp_file:
                temp_file.write(image_content)
                print(f"Image saved to temporary file: {temp_path}")
        except Exception as e:
            print(f"Error saving to temporary file: {str(e)}")
            raise
            
        # Process the image with the predictor, passing the image hash for deterministic results
        print("Processing image with predictor...")
        try:
            # Debug the image hash before passing it
            print(f"Using image hash for prediction: {image_hash}")
            # Make sure we call the method with positional arguments in the correct order
            result = predictor.predict(temp_path, timeframe, image_hash)
            print(f"Prediction result: {result}")
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            raise
        
        # Add image hash to the result
        result['image_hash'] = image_hash
        print("Added image hash to result")
            
        # Store in cache for future requests
        prediction_cache[cache_key] = result
        print("Stored result in cache")
            
        # Clean up the temp file
        try:
            os.unlink(temp_path)
            print("Cleaned up temporary file")
        except Exception as e:
            print(f"Error cleaning up temporary file: {str(e)}")
            
        print("Returning result")
        return result
    except Exception as e:
        print(f"Error in predict_chart: {str(e)}")
        return {"error": str(e)}

# Define models for OHLCV data
class OHLCVData(BaseModel):
    timestamp: str
    open: float
    high: float
    low: float
    close: float
    volume: float

class OHLCVRequest(BaseModel):
    timeframe: str
    chart_data: List[OHLCVData]

@app.post("/api/predict/ohlcv")
async def predict_from_ohlcv(request: OHLCVRequest):
    """Predict from OHLCV data"""
    try:
        # For now, we'll generate a mock prediction for OHLCV data
        # In a real implementation, you would pass this data to your prediction model
        prediction_class = random.choice(["BUY", "SELL", "HOLD"])
        confidence = random.uniform(0.6, 0.9)
        
        # Generate some random patterns
        patterns = []
        all_patterns = ["bullish_engulfing", "bearish_engulfing", "hammer", 
                      "shooting_star", "doji", "morning_star", "evening_star"]
        
        # Select 1-3 random patterns
        num_patterns = random.randint(1, 3)
        for _ in range(num_patterns):
            pattern = random.choice(all_patterns)
            if pattern not in patterns:
                patterns.append(pattern)
        
        # Generate random technical indicators
        indicators = {
            "sma_20": round(random.uniform(90, 110), 2),
            "ema_20": round(random.uniform(90, 110), 2),
            "rsi_14": round(random.uniform(30, 70), 2),
            "macd_line": round(random.uniform(-2, 2), 2),
            "signal_line": round(random.uniform(-2, 2), 2),
            "bb_upper": round(random.uniform(105, 115), 2),
            "bb_lower": round(random.uniform(85, 95), 2),
        }
        
        return {
            "prediction": prediction_class,
            "confidence": confidence,
            "detected_patterns": patterns,
            "technical_indicators": indicators,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
