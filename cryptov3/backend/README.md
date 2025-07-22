# Candlestick Chart Prediction Backend

This is the backend service for the Candlestick Chart Prediction Platform. It's a FastAPI application that processes candlestick chart images and makes predictions using advanced algorithms.

## Setup

1. Create a virtual environment:

```bash
python -m venv venv
```

2. Activate the virtual environment:

- Windows:
```bash
venv\Scripts\activate
```

- Linux/Mac:
```bash
source venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run the server:

```bash
# Standard server with /api/predict endpoint
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# OR run the unified server with all endpoints
python unified_server.py
```

You can also use the batch file to start the unified server on Windows:

```bash
start_unified_server.bat
```

## API Endpoints

### Main API Endpoints

- `GET /` - Health check endpoint
- `GET /api/health` - Extended health check with version and cache info
- `POST /api/predict` - Predict from a candlestick chart image
  - Parameters:
    - `image`: File (JPG/PNG)
    - `timeframe`: string (15m/1h/4h/1d/7d)
  - Response:
    ```json
    {
      "prediction": "BUY",
      "confidence": 0.92,
      "detected_patterns": ["Bullish Engulfing", "Morning Star"],
      "technical_indicators": {
        "RSI": 65,
        "MACD": 0.45,
        "SMA_50": 10450,
        "SMA_200": 10200
      },
      "image_hash": "a1b2c3d4e5f6...",
      "timestamp": "2025-07-22T10:15:30.123456"
    }
    ```

### Advanced Model Endpoint (Unified Server Only)

- `POST /api/imgNewModel` - Advanced analysis for both chart and non-chart images
  - Parameters:
    - `image`: File (JPG/PNG)
    - `timeframe`: string (15m/1h/4h/1d/7d)
    - `predictionInput` (optional): Additional context for prediction
  
  - Response for Chart Images:
    ```json
    {
      "isChart": true,
      "prediction": "BUY",
      "confidence": 0.85,
      "detected_patterns": ["Bullish Engulfing", "Morning Star"],
      "technical_indicators": { ... },
      "image_hash": "a1b2c3d4e5f6...",
      "timestamp": "2025-07-22T10:15:30.123456"
    }
    ```
  
  - Response for Non-Chart Images:
    ```json
    {
      "isChart": false,
      "visual_features": ["High Contrast", "Multiple Colors", "Geometric Shapes"],
      "image_hash": "a1b2c3d4e5f6...",
      "timestamp": "2025-07-22T10:15:30.123456"
    }
    ```

## Algorithms

The backend includes a comprehensive set of algorithms for candlestick chart analysis:

1. **Data Preprocessing** - Handles OHLCV data cleaning, normalization, feature extraction
2. **Technical Indicators** - Calculates SMA, EMA, RSI, MACD, Bollinger Bands, etc.
3. **Candlestick Patterns** - Detects patterns like Doji, Hammer, Engulfing, Morning Star, etc.
4. **Statistical Models** - Implements ARIMA, SARIMA, Kalman filters
5. **Machine Learning** - Uses Random Forest, XGBoost, SVM
6. **Deep Learning** - LSTM, CNN, and hybrid models for time series prediction

All these algorithms are integrated in a modular approach that can process both images and OHLCV data.

## Models

The application uses an integrated model approach with multiple algorithms:

1. **Image-Based Analysis**: 
   - Extracts features from candlestick chart images
   - Recognizes patterns visually
   - Makes predictions based on image characteristics

2. **Rule-Based Systems**:
   - Uses classic candlestick pattern detection rules
   - Applies technical indicator signals
   - Combines signals with confidence weighting

3. **Machine Learning Models** (when trained):
   - Place models in the `models` directory
   - The system automatically detects and uses available models
   - Models enhance prediction accuracy when available

To use your own models:

```python
predictor = CandlestickPredictor("path/to/your/models_directory")
```

## Image Hash-Based Determinism

To ensure consistent results for the same image, we:

1. Generate a hash of the image content
2. Use this hash to seed random number generators
3. Cache results based on the image hash and timeframe
4. Return the same prediction for identical images

This makes the API's predictions reproducible and consistent.

## Testing and Debugging

The project includes comprehensive testing tools:

1. **test_api.py** - Tests the `/api/predict` endpoint with images
2. **test_unified_api.py** - Tests both endpoints in the unified server
3. **test_consistency.py** - Verifies that image hashing produces consistent results

Run the tests with:

```bash
python test_unified_api.py
```

## Troubleshooting

Common issues:

1. **404 Not Found for `/api/imgNewModel`**: Make sure you're running `unified_server.py` instead of just `main.py`.
2. **Inconsistent Results**: Verify the image hash is being correctly used to seed random generators.
3. **Module Not Found Errors**: Install all dependencies with `pip install -r requirements.txt`.
4. **TensorFlow Errors**: The system includes a fallback for environments without TensorFlow.
