# Candlestick Chart Prediction Models

This directory is for storing trained machine learning and deep learning models for cryptocurrency price prediction.

## Model Organization

Models are organized by symbol and timeframe:

```
models/
  ├── BTC-USD/
  │   ├── 1d/
  │   │   └── btc_usd_1d_20230415_121530_model.pkl
  │   │   └── btc_usd_1d_20230415_121530_metadata.json
  │   ├── 4h/
  │   │   └── ...
  ├── ETH-USD/
  │   ├── 1d/
  │   │   └── ...
  └── ...
```

## Model Types

The system supports several types of models:

1. **Deep Learning Models**:
   - LSTM models (.h5 files)
   - CNN models (.h5 files)
   - CNN-LSTM hybrid models (.h5 files)
   - Associated metadata files (.json)

2. **Traditional ML Models**:
   - Random Forest, XGBoost, SVM, etc. (.pkl files)
   - Feature scalers and preprocessing info

## Training New Models

Models can be trained using the provided examples:

### Using the Command Line Tool

```bash
python ../examples/train_model.py --file /path/to/data.csv --symbol BTC-USD --timeframe 1d --output ./BTC-USD/1d --visualize
```

### Using Python Code

```python
from algorithms.integrated_model import IntegratedModel
import pandas as pd

# Load data
df = pd.read_csv('your_ohlcv_data.csv')

# Initialize model
model = IntegratedModel()

# Train models and save to this directory
results = model.train_models(df, save_dir='./models')
```

## Using Models in Production

To load and use a saved model:

```python
import joblib
from pathlib import Path

# Load model
model_path = Path("./BTC-USD/1d/btc_usd_1d_20230415_121530_model.pkl")
model = joblib.load(model_path)

# Prepare data with the same features used during training
# ...

# Make prediction
prediction = model.predict(features)
```

## Model Selection

When no models are available, the system falls back to rule-based algorithms and pattern recognition for predictions. As models are trained and added to this directory, the system will automatically use them for improved accuracy.

## Model Performance

Models are evaluated using multiple metrics:
- R² (coefficient of determination)
- MAE (Mean Absolute Error)
- MSE (Mean Squared Error)
- RMSE (Root Mean Squared Error)

For classification models (predicting price direction):
- Accuracy
- Precision
- Recall
- F1 score

## Recommended Models

For optimal performance, consider training and adding:

- An LSTM model for sequence prediction (for each symbol/timeframe)
- A Random Forest model for price prediction (for each symbol/timeframe)
- An XGBoost model for direction classification (for each symbol/timeframe)

## Example Commands

### Train a Basic Price Prediction Model
```bash
python ../examples/train_model.py --file data/BTC-USD.csv --symbol BTC-USD --target next_close --horizon 1
```

### Train a Direction Prediction Model
```bash
python ../examples/train_model.py --file data/ETH-USD.csv --symbol ETH-USD --target direction --horizon 3
```
