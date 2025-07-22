# Candlestick Chart Prediction Algorithms

This directory contains various algorithms for analyzing and predicting candlestick chart patterns.

## Algorithm Categories

1. **Data Preprocessing** (`preprocessing.py`):
   - OHLCV data loading and cleaning
   - Time series resampling and normalization
   - Log returns computation
   - Rolling statistics calculation (mean, std, skew, kurtosis)
   - VWAP & TWAP calculation
   - Image feature extraction

2. **Technical Indicators** (`technical_indicators.py`):
   - Trend indicators (SMA, EMA, DEMA, TEMA)
   - Momentum indicators (RSI, Stochastic Oscillator, ROC)
   - Trend-momentum indicators (MACD)
   - Volatility indicators (Bollinger Bands, ATR, Donchian)
   - Volume indicators (OBV, VPT, MFI)

3. **Candlestick Patterns** (`candlestick_patterns.py`):
   - Single-bar patterns (Doji, Hammer, Shooting Star, Spinning Top, Marubozu)
   - Multi-bar patterns (Engulfing, Harami, Morning/Evening Star, Tweezer Tops/Bottoms)
   - Pattern detection algorithms
   - Image-based pattern recognition

4. **Statistical & Time-Series Models** (`time_series_models.py`):
   - ARIMA / SARIMA models
   - Exponential smoothing
   - Seasonal decomposition
   - Kalman filtering
   - Optimal model parameter selection

5. **Machine Learning Models** (`ml_models.py`):
   - Random Forests
   - Gradient Boosting (XGBoost)
   - SVMs
   - k-Nearest Neighbors
   - Direction prediction models

6. **Deep Learning Models** (`dl_models.py`):
   - LSTM/GRU for sequence modeling
   - 1D-CNNs for pattern recognition
   - Hybrid CNN-LSTM architectures
   - Model saving and loading utilities

7. **Integrated Model** (`integrated_model.py`):
   - Combined algorithm approach
   - Ensemble predictions
   - End-to-end prediction pipeline

## Simplified Demo

A simplified demo has been created to showcase the core functionality without requiring all dependencies. The demo can be run using the `run_demo.bat` file in the root directory. This simplified demo includes:

- Data preprocessing examples
- Technical indicator calculations
- Candlestick pattern detection
- Time series modeling with ARIMA

The more complex machine learning and deep learning models are excluded from this simplified demo to reduce dependencies.
   - Image and OHLCV data integration

## Usage

The algorithms are used by the `CandlestickPredictor` class in the services directory. The integrated model combines multiple approaches for the best prediction results.

Example usage:

```python
from algorithms.integrated_model import IntegratedModel

# Initialize model
model = IntegratedModel()

# Predict from image
result = model.predict_from_image("chart.png", timeframe="1h")
print(result)
```

## Implementation Notes

- The algorithms are designed to work with both image data and OHLCV time series
- Pattern recognition can work on chart images without requiring raw price data
- When OHLCV data is available, more sophisticated models can be employed
- The ensemble approach combines multiple algorithms for higher accuracy
