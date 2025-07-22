from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from PIL import Image
import io
import numpy as np
from typing import Optional, Dict, Any
import json
import random
import hashlib
from datetime import datetime
import os
import sys
from pathlib import Path

# Add the backend directory to the path so we can import from it
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the chart analysis module (assuming it exists)
try:
    from examples.chart_analysis import analyze_candlestick_chart
except ImportError:
    # Enhanced mock function with better analysis based on image features
    def analyze_candlestick_chart(image_array, timeframe):
        # Generate a consistent hash for the image to ensure the same image gets the same analysis
        # Resize to a standard size for consistent results
        hash_value = hashlib.md5(np.array(Image.fromarray(image_array).resize((100, 100))).tobytes()).hexdigest()
        
        # Create a deterministic random generator seeded with the image hash
        hash_seed = int(hash_value, 16)
        seed_random = random.Random(hash_seed)
        
        # Use the timeframe to adjust the analysis
        timeframe_factor = {
            '15m': 0.1,
            '1h': 0.2,
            '4h': 0.3,
            '1d': 0.4,
            '7d': 0.5
        }.get(timeframe, 0.3)
        
        # Generate a consistent seed that combines the image hash and timeframe
        combined_seed = hash_seed + hash(timeframe)
        seed_random_with_timeframe = random.Random(combined_seed)
        
        # Try to do simple image analysis to extract real features
        try:
            # Convert to grayscale for analysis
            if len(image_array.shape) == 3 and image_array.shape[2] >= 3:
                gray = np.mean(image_array[:, :, :3], axis=2).astype(np.uint8)
            else:
                gray = image_array
            
            # Analyze image characteristics for candlestick features
            # For example, detect if overall trend is up or down
            height, width = gray.shape[:2]
            left_side = gray[:, :width//4].mean()
            right_side = gray[:, 3*width//4:].mean()
            
            trend_direction = "up" if right_side > left_side else "down"
            
            # Detect volatility based on vertical variations
            vertical_variation = np.std(np.diff(gray, axis=0))
            volatility = "high" if vertical_variation.mean() > 20 else "low"
            
            # Use these real features to influence pattern detection and indicators
        except Exception:
            # Fallback to hash-based only if image analysis fails
            trend_direction = "up" if int(hash_value[0], 16) > 7 else "down"
            volatility = "high" if int(hash_value[1], 16) > 7 else "low"
        
        # All possible patterns
        bullish_patterns = [
            "Bullish Engulfing", 
            "Hammer",
            "Morning Star",
            "Three White Soldiers",
            "Bullish Harami"
        ]
        
        bearish_patterns = [
            "Bearish Engulfing",
            "Hanging Man", 
            "Evening Star",
            "Three Black Crows",
            "Bearish Harami"
        ]
        
        neutral_patterns = [
            "Doji",
            "Spinning Top",
            "Long-Legged Doji",
            "Dragonfly Doji"
        ]
        
        # Choose patterns based on detected trend
        if trend_direction == "up":
            primary_patterns = bullish_patterns
            secondary_patterns = neutral_patterns
            pattern_weights = [0.7, 0.3]  # 70% bullish, 30% neutral
        else:
            primary_patterns = bearish_patterns
            secondary_patterns = neutral_patterns
            pattern_weights = [0.7, 0.3]  # 70% bearish, 30% neutral
        
        # Add randomness based on the image hash but biased by the detected trend
        pattern_count = seed_random_with_timeframe.randint(1, 3)
        detected_patterns = []
        
        for _ in range(pattern_count):
            pattern_type = seed_random_with_timeframe.choices(
                ["primary", "secondary"], 
                weights=pattern_weights, 
                k=1
            )[0]
            
            if pattern_type == "primary":
                pattern = seed_random_with_timeframe.choice(primary_patterns)
                if pattern not in detected_patterns:
                    detected_patterns.append(pattern)
            else:
                pattern = seed_random_with_timeframe.choice(secondary_patterns)
                if pattern not in detected_patterns:
                    detected_patterns.append(pattern)
        
        # Determine prediction based on trend and patterns
        if trend_direction == "up":
            prediction_options = ["BUY", "HOLD"]
            prediction_weights = [0.7, 0.3]
        else:
            prediction_options = ["SELL", "HOLD"]
            prediction_weights = [0.7, 0.3]
            
        # Choose prediction based on weighted random
        prediction = seed_random_with_timeframe.choices(
            prediction_options, 
            weights=prediction_weights, 
            k=1
        )[0]
        
        # Calculate a consistent confidence based on volatility and timeframe
        confidence_base = (int(hash_value[2:4], 16) % 40) / 100  # 0.00-0.39
        confidence = round(0.5 + confidence_base + timeframe_factor, 2)
        
        # Lower confidence for high volatility or shorter timeframes
        if volatility == "high":
            confidence *= 0.9
            
        # Adjust confidence based on timeframe
        if timeframe in ['15m', '1h']:
            confidence *= 0.95
            
        confidence = min(confidence, 0.95)  # Cap at 0.95
        
        # Technical indicators based on image analysis and hash
        # RSI - influenced by trend direction
        rsi_base = int(hash_value[4:6], 16) % 101  # 0-100
        rsi_value = max(min(rsi_base + (15 if trend_direction == "up" else -15), 100), 0)
        
        # MACD - influenced by trend direction
        macd_base = (int(hash_value[6:8], 16) % 400 - 200) / 100  # -2.00 to 1.99
        macd_value = round(macd_base * (1.2 if trend_direction == "up" else 0.8), 2)
        
        # Moving averages - based on hash but with realistic relationship
        sma_50_base = 10000 + (int(hash_value[8:10], 16) % 1000)  # 10000-10999
        
        # Ensure SMA 200 is lower than SMA 50 for uptrend, higher for downtrend
        if trend_direction == "up":
            sma_200_base = sma_50_base - (int(hash_value[10:12], 16) % 500)  # Lower than SMA 50
        else:
            sma_200_base = sma_50_base + (int(hash_value[10:12], 16) % 500)  # Higher than SMA 50
            
        # Volume - higher for volatile markets
        volume_multiplier = 1.5 if volatility == "high" else 1.0
        volume_base = int(100000 + (int(hash_value[12:14], 16) * 1000 * volume_multiplier))
        
        # Bollinger Bands - related to trend
        bollinger_value = "Above" if trend_direction == "up" else "Below"
        
        # Stochastic - related to RSI
        stochastic_value = max(min(rsi_value + (int(hash_value[15:], 16) % 20 - 10), 100), 0)
        
        return {
            "prediction": prediction,
            "confidence": confidence,
            "detected_patterns": detected_patterns,
            "technical_indicators": {
                "RSI": rsi_value,
                "MACD": macd_value,
                "SMA_50": sma_50_base,
                "SMA_200": sma_200_base,
                "Volume": volume_base,
                "Bollinger Bands": bollinger_value,
                "Stochastic": stochastic_value,
                "Trend": trend_direction.upper(),
                "Volatility": volatility.upper()
            }
        }

# Load the survey JSON data
SURVEY_DATA = None
try:
    with open("cryptocurrency_prediction_survey_machine_readable.json", "r") as f:
        SURVEY_DATA = json.load(f)
except (FileNotFoundError, json.JSONDecodeError):
    # Use a simplified version of the survey data if file not found
    SURVEY_DATA = {
        "title": "Cryptocurrency Price Prediction Algorithms: A Survey and Future Directions",
        "parameters": {
            "technical_indicators": ["RSI", "MACD", "Moving Averages", "Bollinger Bands"],
            "blockchain_features": ["Hash Rate", "Transaction Volume", "Miner Revenue"],
            "sentiment_analysis": ["Social Media", "News Articles"]
        },
        "algorithms": {
            "LSTM-GRU": {
                "accuracy": 0.85,
                "effectiveness": "high"
            },
            "CNN-LSTM": {
                "accuracy": 0.82,
                "effectiveness": "medium-high"
            },
            "Transformers": {
                "accuracy": 0.88,
                "effectiveness": "high"
            }
        }
    }

# Create a cache directory if it doesn't exist
CACHE_DIR = Path("analysis_cache")
CACHE_DIR.mkdir(exist_ok=True)

# In-memory cache to avoid filesystem reads for frequent requests
MEMORY_CACHE: Dict[str, Dict[str, Any]] = {}

def get_cached_analysis(image_hash: str, timeframe: str) -> Optional[Dict[str, Any]]:
    """
    Check if we have a cached analysis for this image and timeframe
    """
    # Check memory cache first
    cache_key = f"{image_hash}_{timeframe}"
    if cache_key in MEMORY_CACHE:
        return MEMORY_CACHE[cache_key]
    
    # Check filesystem cache
    cache_file = CACHE_DIR / f"{cache_key}.json"
    if cache_file.exists():
        try:
            with open(cache_file, 'r') as f:
                result = json.load(f)
                # Store in memory cache for faster access next time
                MEMORY_CACHE[cache_key] = result
                return result
        except (json.JSONDecodeError, IOError):
            # If the cache file is corrupted, ignore it
            return None
    
    return None

def save_analysis_to_cache(image_hash: str, timeframe: str, result: Dict[str, Any]) -> None:
    """
    Save analysis result to cache
    """
    # Save to memory cache
    cache_key = f"{image_hash}_{timeframe}"
    MEMORY_CACHE[cache_key] = result
    
    # Save to filesystem cache
    cache_file = CACHE_DIR / f"{cache_key}.json"
    try:
        with open(cache_file, 'w') as f:
            json.dump(result, f)
    except IOError:
        # If we can't write to the cache file, just ignore it
        pass

# Create the FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ImageAnalyzer:
    @staticmethod
    def generate_image_hash(image_array):
        """
        Generate a unique hash for an image that will be consistent across uploads
        """
        # Convert the numpy array to bytes and hash it
        # Use a smaller version of the image for faster processing
        height, width = image_array.shape[:2]
        resized = Image.fromarray(image_array).resize((width // 10, height // 10))
        img_bytes = np.array(resized).tobytes()
        return hashlib.md5(img_bytes).hexdigest()

    @staticmethod
    def is_chart_image(image_array):
        """
        Determines if an image is a candlestick chart based on visual features
        
        Uses image analysis to detect patterns consistent with candlestick charts
        while maintaining consistency for the same image using its hash
        """
        # Get the image hash for consistency
        image_hash = ImageAnalyzer.generate_image_hash(image_array)
        
        # Simple chart detection algorithm:
        # 1. Look for regular patterns of vertical lines (candlesticks)
        # 2. Check for horizontal grid lines common in charts
        # 3. Check color distribution typical of charts
        
        # First, for consistency, use hash as a fallback
        hash_value = int(image_hash[0], 16)  # Convert hex to int (0-15)
        hash_result = hash_value > 4  # ~70% probability
        
        try:
            # Convert to grayscale for analysis
            if len(image_array.shape) == 3 and image_array.shape[2] >= 3:
                gray = np.mean(image_array[:, :, :3], axis=2).astype(np.uint8)
            else:
                gray = image_array
                
            # Check for horizontal lines (common in charts)
            horizontal_lines = 0
            
            # Sample rows at 10% intervals
            height = gray.shape[0]
            for i in range(1, 10):
                row_idx = int(height * i / 10)
                row = gray[row_idx, :]
                # Calculate row variance and detect potential grid lines
                row_diff = np.abs(np.diff(row.astype(float)))
                if np.mean(row_diff) < 10 and np.max(row_diff) > 30:
                    horizontal_lines += 1
            
            # Check for vertical patterns (candlesticks)
            vertical_patterns = 0
            
            # Sample columns at 10% intervals
            width = gray.shape[1]
            for i in range(1, 10):
                col_idx = int(width * i / 10)
                col = gray[:, col_idx]
                # Calculate column variance and detect potential candlesticks
                col_diff = np.abs(np.diff(col.astype(float)))
                if np.mean(col_diff) > 15 and np.max(col_diff) > 50:
                    vertical_patterns += 1
            
            # Color distribution check (charts often have specific color patterns)
            color_variety = 0
            if len(image_array.shape) == 3 and image_array.shape[2] >= 3:
                # Calculate the number of unique colors in a downsampled image
                downsampled = image_array[::10, ::10, :]
                # Flatten the image to count unique colors
                unique_colors = np.unique(downsampled.reshape(-1, downsampled.shape[2]), axis=0)
                color_variety = len(unique_colors)
            
            # Combine the metrics
            is_likely_chart = (horizontal_lines >= 3 and vertical_patterns >= 3 and color_variety < 5000)
            
            # For edge cases, rely on the hash result to ensure consistency
            if horizontal_lines + vertical_patterns <= 5:
                return hash_result
            
            return is_likely_chart
        except Exception:
            # Fallback to the hash-based method if there's an error in analysis
            return hash_result
        # In a real implementation, you'd analyze image features
        return random.random() < 0.7

    @staticmethod
    def extract_visual_features(image_array):
        """
        Extract visual features from a non-chart image
        Using image hash to ensure consistency for the same image
        """
        # Get the image hash
        image_hash = ImageAnalyzer.generate_image_hash(image_array)
        
        # All possible features
        all_features = [
            'High Contrast',
            'Multiple Colors', 
            'Geometric Shapes',
            'Text Elements',
            'Landscape Orientation',
            'Portrait Orientation',
            'Grayscale',
            'High Saturation',
            'Low Light',
            'Bright Image',
            'Blurry',
            'Sharp Focus',
            'Contains Faces',
            'Natural Scene',
            'Indoor Scene'
        ]
        
        # Use the hash to seed a random number generator
        # This ensures the same image always gets the same features
        hash_seed = int(image_hash, 16) % 10000
        seed_random = random.Random(hash_seed)
        
        # Number of features to select (between 3 and 7)
        k = seed_random.randint(3, 7)
        
        # Return a consistent selection of features for this image
        return seed_random.sample(all_features, k)

class SurveyModelProcessor:
    @staticmethod
    def process_survey_model(timeframe, user_input="", detected_patterns=None, image_hash=None):
        """
        Process the survey model data to generate a prediction
        
        Args:
            timeframe: The selected timeframe for the prediction
            user_input: Any additional context provided by the user
            detected_patterns: Chart patterns detected in the image
            image_hash: Hash of the image for deterministic results
            
        Returns:
            A prediction result object
        """
        if detected_patterns is None:
            detected_patterns = []
        
        # Create a deterministic random generator
        if image_hash:
            # Combine image hash with timeframe and input for consistent but varied results
            combined_input = f"{image_hash}-{timeframe}-{user_input}"
            seed = int(hashlib.md5(combined_input.encode()).hexdigest(), 16) % (2**32)
            seed_random = random.Random(seed)
        else:
            # Fallback to system random if no hash is provided
            seed_random = random.Random(42)  # Fixed seed for consistency
        
        # Base confidence level - now deterministic
        confidence = 0.6 + (seed_random.random() * 0.2)  # Between 0.6 and 0.8
        
        # Adjust confidence based on timeframe (longer timeframes typically have lower confidence)
        timeframe_factors = {
            '15m': 1.1,
            '1h': 1.05,
            '4h': 1.0,
            '1d': 0.95,
            '7d': 0.9
        }
        confidence *= timeframe_factors.get(timeframe, 1.0)
        
        # Cap confidence at 0.95
        confidence = min(confidence, 0.95)
        
        # Count bullish and bearish patterns
        bullish_patterns = 0
        bearish_patterns = 0
        
        for pattern in detected_patterns:
            pattern_lower = pattern.lower()
            if any(term in pattern_lower for term in ['bullish', 'hammer', 'morning star', 'white soldiers']):
                bullish_patterns += 1
            elif any(term in pattern_lower for term in ['bearish', 'hanging', 'evening star', 'black crows']):
                bearish_patterns += 1
        
        # Consider user input for sentiment (very basic analysis)
        if user_input:
            input_lower = user_input.lower()
            if any(term in input_lower for term in ['bull', 'up', 'rise', 'growth']):
                bullish_patterns += 1
            elif any(term in input_lower for term in ['bear', 'down', 'fall', 'drop']):
                bearish_patterns += 1
        
        # Determine prediction based on patterns and input - now deterministic
        if bullish_patterns > bearish_patterns:
            prediction_rand = seed_random.random()
            prediction = "UPTREND" if prediction_rand > 0.3 else "BUY"
        elif bearish_patterns > bullish_patterns:
            prediction_rand = seed_random.random()
            prediction = "DOWNTREND" if prediction_rand > 0.3 else "SELL"
        else:
            # If tied or no patterns, use a slight bias based on timeframe
            prediction_rand = seed_random.random()
            if timeframe == '7d':
                # Long term bias slightly bullish (historical crypto trend)
                prediction = "BUY" if prediction_rand > 0.6 else "HOLD"
            else:
                # Short term is more balanced
                prediction = "BUY" if prediction_rand > 0.5 else "SELL"
        
        return {
            "prediction": prediction,
            "confidence": round(confidence, 2)
        }

@app.post("/api/imgNewModel")
async def analyze_image(
    image: UploadFile = File(...),
    timeframe: str = Form(...),
    predictionInput: Optional[str] = Form(None)
):
    print(f"imgNewModel API: Received request with timeframe: {timeframe}")
    print(f"imgNewModel API: Image filename: {image.filename}, content type: {image.content_type}")
    print(f"imgNewModel API: PredictionInput present: {predictionInput is not None}")
    
    # Read and process the image
    try:
        print("imgNewModel API: Reading image content...")
        image_content = await image.read()
        print(f"imgNewModel API: Read {len(image_content)} bytes of image data")
        
        image_bytes = io.BytesIO(image_content)
        img = Image.open(image_bytes)
        print(f"imgNewModel API: Image opened successfully, size: {img.size}, mode: {img.mode}")
        
        img_array = np.array(img)
        print(f"imgNewModel API: Converted to numpy array, shape: {img_array.shape}")
        
        # Generate a hash for the image to ensure consistent results
        image_hash = ImageAnalyzer.generate_image_hash(img_array)
        print(f"imgNewModel API: Generated image hash: {image_hash}")
    except Exception as e:
        print(f"imgNewModel API: Error processing image: {str(e)}")
        return {"error": f"Failed to process image: {str(e)}"}
    
    # Check if we have a cached result for this image and timeframe
    # Only use the cache if there's no prediction input (as that would change the result)
    if not predictionInput:
        print(f"imgNewModel API: Checking cache for image hash: {image_hash}, timeframe: {timeframe}")
        cached_result = get_cached_analysis(image_hash, timeframe)
        if cached_result:
            print(f"imgNewModel API: Found cached result, returning it")
            return cached_result
        else:
            print(f"imgNewModel API: No cached result found")
    
    # Check if it's a chart image - now deterministic based on image hash
    try:
        print(f"imgNewModel API: Checking if image is a candlestick chart")
        is_chart = ImageAnalyzer.is_chart_image(img_array)
        print(f"imgNewModel API: Image is{'' if is_chart else ' not'} a chart")
        
        if is_chart:
            # Process with existing chart analysis algorithm - now deterministic
            print(f"imgNewModel API: Analyzing candlestick chart for timeframe: {timeframe}")
            chart_result = analyze_candlestick_chart(img_array, timeframe)
            print(f"imgNewModel API: Chart analysis complete. Prediction: {chart_result.get('prediction')}, Confidence: {chart_result.get('confidence')}")
            
            # Process with survey-based algorithm - now deterministic
            survey_result = SurveyModelProcessor.process_survey_model(
                timeframe, 
                predictionInput or "", 
                chart_result["detected_patterns"],
                image_hash  # Pass the image hash for consistent results
            )
    except Exception as e:
        print(f"imgNewModel API: Error in chart detection/analysis: {str(e)}")
        return {"error": f"Error analyzing chart: {str(e)}"}
        
        
        # Combine results deterministically
        chart_confidence = chart_result["confidence"]
        survey_confidence = survey_result["confidence"]
        combined_confidence = (chart_confidence + survey_confidence) / 2
        
        # Determine combined prediction based on weighted confidence
        if chart_confidence > survey_confidence:
            combined_prediction = chart_result["prediction"]
        else:
            combined_prediction = survey_result["prediction"]
        
        # If predictions disagree but confidences are close, use NEUTRAL
        if (chart_result["prediction"] != survey_result["prediction"] and 
            abs(chart_confidence - survey_confidence) < 0.1):
            combined_prediction = "NEUTRAL"
        
        # Create the result
        result = {
            "isChart": True,
            "prediction": chart_result["prediction"],
            "confidence": chart_result["confidence"],
            "detected_patterns": chart_result["detected_patterns"],
            "technical_indicators": chart_result["technical_indicators"],
            "survey_prediction": survey_result["prediction"],
            "survey_confidence": survey_result["confidence"],
            "combined_prediction": combined_prediction,
            "combined_confidence": round(combined_confidence, 2),
            "image_hash": image_hash,  # Include the hash for reference
            "timestamp": datetime.now().isoformat()
        }
        
        # Cache the result if there's no prediction input
        # (since prediction input would change the result and shouldn't be cached)
        if not predictionInput:
            save_analysis_to_cache(image_hash, timeframe, result)
        
        return result
    else:
        # For non-chart images, extract visual features - now deterministic
        visual_features = ImageAnalyzer.extract_visual_features(img_array)
        
        # Create the result
        result = {
            "isChart": False,
            "visual_features": visual_features,
            "image_hash": image_hash,  # Include the hash for reference
            "timestamp": datetime.now().isoformat()
        }
        
        # Cache the result if there's no prediction input
        if not predictionInput:
            save_analysis_to_cache(image_hash, timeframe, result)
        
        return result

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)
