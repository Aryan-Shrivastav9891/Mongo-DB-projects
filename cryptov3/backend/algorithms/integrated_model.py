import pandas as pd
import numpy as np
import cv2
from typing import Dict, List, Any, Optional, Union, Tuple
import os
import logging
import json
from .preprocessing import DataPreprocessor
from .technical_indicators import TechnicalIndicators
from .candlestick_patterns import CandlestickPatterns
from .time_series_models import TimeSeriesModels
from .ml_models import MachineLearningModels
# Removing DeepLearningModels import as it requires TensorFlow
# from .dl_models import DeepLearningModels

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IntegratedModel:
    """
    Integrated model that combines multiple algorithms for candlestick chart prediction
    """
    
    def __init__(self, models_dir: str = None):
        """
        Initialize the integrated model
        
        Parameters:
        -----------
        models_dir : str
            Directory containing trained models
        """
        self.models_dir = models_dir
        
        # Initialize components
        self.preprocessor = DataPreprocessor()
        self.indicators = TechnicalIndicators()
        self.patterns = CandlestickPatterns()
        self.ts_models = TimeSeriesModels()
        self.ml_models = MachineLearningModels()
        # Removing DeepLearningModels as it requires TensorFlow
        # self.dl_models = DeepLearningModels()
        
        # Load pre-trained models if available
        self._load_models()
    
    def _load_models(self) -> None:
        """Load pre-trained models if available"""
        if not self.models_dir or not os.path.exists(self.models_dir):
            logger.warning("No models directory provided or directory does not exist.")
            return
        
        try:
            # Load ML models (h5 deep learning models are skipped)
            for filename in os.listdir(self.models_dir):
                if filename.endswith('.pkl'):
                    logger.info(f"Found ML model: {filename}")
                # Deep learning models are skipped
                # if filename.endswith('.h5'):
                #     model_name = filename[:-3]  # Remove .h5 extension
                #     self.dl_models.load_model(model_name, self.models_dir)
                #     logger.info(f"Loaded deep learning model: {model_name}")
        except Exception as e:
            logger.error(f"Error loading models: {e}")
    
    def preprocess_image(self, image_path: str) -> Dict[str, Any]:
        """
        Preprocess a candlestick chart image for analysis
        
        Parameters:
        -----------
        image_path : str
            Path to the image file
            
        Returns:
        --------
        Dict[str, Any]
            Preprocessed image and features
        """
        try:
            print(f"IntegratedModel: Preprocessing image from {image_path}")
            if not os.path.exists(image_path):
                print(f"IntegratedModel: File does not exist: {image_path}")
                raise FileNotFoundError(f"File does not exist: {image_path}")
                
            # Read image
            print(f"IntegratedModel: Reading image with OpenCV")
            img = cv2.imread(image_path)
            if img is None:
                print(f"IntegratedModel: Could not read image at {image_path}")
                raise ValueError(f"Could not read image at {image_path}")
            
            print(f"IntegratedModel: Image read successfully, shape: {img.shape}")
            
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            print(f"IntegratedModel: Converted to grayscale, shape: {gray.shape}")
            
            # Resize to standard size
            resized = cv2.resize(gray, (256, 256))
            print(f"IntegratedModel: Resized image, shape: {resized.shape}")
            
            # Normalize pixel values to [0, 1]
            normalized = resized / 255.0
            
            # Extract features from image
            print(f"IntegratedModel: Extracting features from image")
            features = self.preprocessor.extract_features_from_image(normalized)
            print(f"IntegratedModel: Features extracted: {list(features.keys())}")
            
            return {
                'image': normalized,
                'features': features
            }
        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            raise
    
    def detect_patterns_from_image(self, image_path: str) -> List[str]:
        """
        Detect candlestick patterns from an image
        
        Parameters:
        -----------
        image_path : str
            Path to the image file
            
        Returns:
        --------
        List[str]
            Detected patterns
        """
        try:
            # Preprocess the image
            result = self.preprocess_image(image_path)
            
            # Detect patterns from image features
            patterns = self.patterns.detect_patterns_from_image(result['features'])
            
            return patterns
        except Exception as e:
            logger.error(f"Error detecting patterns from image: {e}")
            raise
    
    def predict_from_image(self, image_path: str, timeframe: str, image_hash: str = None) -> Dict[str, Any]:
        """
        Make a prediction from a candlestick chart image
        
        Parameters:
        -----------
        image_path : str
            Path to the image file
        timeframe : str
            Timeframe of the chart
        image_hash : str
            Hash of the image to ensure deterministic results
            
        Returns:
        --------
        Dict[str, Any]
            Prediction results
        """
        try:
            # If image hash is provided, set a fixed seed for reproducible results
            if image_hash:
                # Create a deterministic seed from the image hash
                # This ensures same image always produces same results
                seed = int(image_hash[:8], 16) % (2**32 - 1)  # Use first 8 chars of hash as hex number
                logger.info(f"Setting seed for reproducible results: {image_hash[:8]} -> {seed}")
                np.random.seed(seed)
            
            # Preprocess the image
            result = self.preprocess_image(image_path)
            
            # Detect patterns
            patterns = self.patterns.detect_patterns_from_image(result['features'])
            
            # Predict based on image features
            prediction, confidence = self._predict_from_features(result['features'], timeframe)
            
            # Prepare result
            return {
                'prediction': prediction,
                'confidence': confidence,
                'patterns': patterns,
                'timeframe': timeframe
            }
        except Exception as e:
            logger.error(f"Error making prediction from image: {e}")
            raise
    
    def _predict_from_features(self, features: Dict[str, float], timeframe: str) -> Tuple[str, float]:
        """
        Make a prediction based on image features
        
        Parameters:
        -----------
        features : Dict[str, float]
            Image features
        timeframe : str
            Timeframe of the chart
            
        Returns:
        --------
        Tuple[str, float]
            (prediction, confidence)
        """
        # Extract key features
        mean_val = features['mean']
        std_val = features['std']
        contrast = features['contrast']
        grad_mean = features['grad_mean']
        
        # Apply heuristic rules based on timeframe
        # This is a placeholder implementation
        if timeframe in ['15m', '1h']:  # Shorter timeframes
            if mean_val > 0.5 and std_val < 0.2:
                # Brighter, low contrast image - suggest BUY
                return 'BUY', 0.65 + (mean_val - 0.5)
            elif mean_val < 0.4:
                # Darker image - suggest SELL
                return 'SELL', 0.70 + (0.4 - mean_val)
            else:
                # Otherwise suggest HOLD
                return 'HOLD', 0.60 + std_val
                
        elif timeframe in ['4h', '1d', '7d']:  # Longer timeframes
            if contrast > 0.7:  # High contrast
                if mean_val > 0.5:
                    return 'BUY', 0.75 + (mean_val - 0.5) / 2
                else:
                    return 'SELL', 0.80 - (mean_val / 2)
            else:  # Low contrast
                return 'HOLD', 0.65 + std_val
        
        # Default fallback
        return 'HOLD', 0.50
    
    def analyze_ohlcv_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze OHLCV data to detect patterns and calculate indicators
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with OHLCV data
            
        Returns:
        --------
        Dict[str, Any]
            Analysis results
        """
        try:
            # Calculate technical indicators
            indicators_df = self.indicators.calculate_all_indicators(df)
            
            # Detect candlestick patterns
            patterns_df = self.patterns.analyze_all_patterns(df)
            
            # Combine results
            result_df = pd.concat([indicators_df, patterns_df], axis=1)
            
            # Get the most recent patterns
            recent_patterns = []
            for col in patterns_df.columns:
                if patterns_df.iloc[-1][col]:
                    recent_patterns.append(col)
            
            # Make time series prediction
            next_bar_prediction = self.ts_models.predict_next_bar(
                df, model_type='arima', features=['close']
            )
            
            return {
                'indicators': indicators_df.tail(5).to_dict('records'),
                'patterns': recent_patterns,
                'prediction': next_bar_prediction
            }
        except Exception as e:
            logger.error(f"Error analyzing OHLCV data: {e}")
            raise
    
    def train_models(self, df: pd.DataFrame, save_dir: str = None) -> Dict[str, Any]:
        """
        Train multiple models on OHLCV data
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with OHLCV data
        save_dir : str
            Directory to save trained models
            
        Returns:
        --------
        Dict[str, Any]
            Training results
        """
        try:
            # Calculate technical indicators
            indicators_df = self.indicators.calculate_all_indicators(df)
            
            # Train models
            results = {}
            
            # Random Forest model
            rf_result = self.ml_models.train_random_forest(
                indicators_df, target_col='close', classification=False
            )
            results['random_forest'] = {
                'model_name': rf_result['model_name'],
                'metrics': rf_result['metrics']
            }
            
            # XGBoost model
            xgb_result = self.ml_models.train_xgboost(
                indicators_df, target_col='close', classification=False
            )
            results['xgboost'] = {
                'model_name': xgb_result['model_name'],
                'metrics': xgb_result['metrics']
            }
            
            # Direction prediction model
            direction_result = self.ml_models.train_direction_model(
                indicators_df, window_size=10, model_type='xgboost'
            )
            results['direction'] = {
                'model_name': direction_result['model_name'],
                'metrics': direction_result['metrics']
            }
            
            # LSTM model disabled as it requires TensorFlow
            # if len(df) >= 100:
            #     lstm_result = self.dl_models.train_lstm_model(
            #         indicators_df, target_col='close', sequence_length=10, forecast_horizon=1
            #     )
            #     results['lstm'] = {
            #         'model_name': lstm_result['model_name'],
            #         'metrics': lstm_result['metrics']
            #     }
            #     
            #     # Save deep learning model if save_dir provided
            #     if save_dir:
            #         os.makedirs(save_dir, exist_ok=True)
            #         model_path = self.dl_models.save_model(lstm_result['model_name'], save_dir)
            #         results['lstm']['model_path'] = model_path
            
            return results
        except Exception as e:
            logger.error(f"Error training models: {e}")
            raise

    def ensemble_predict(self, image_path: str, timeframe: str, ohlcv_data: pd.DataFrame = None) -> Dict[str, Any]:
        """
        Make a prediction using an ensemble of models
        
        Parameters:
        -----------
        image_path : str
            Path to the image file
        timeframe : str
            Timeframe of the chart
        ohlcv_data : pd.DataFrame
            Optional OHLCV data for additional analysis
            
        Returns:
        --------
        Dict[str, Any]
            Prediction results
        """
        try:
            # Initialize predictions list
            predictions = []
            confidences = []
            
            # Get image-based prediction
            image_result = self.predict_from_image(image_path, timeframe)
            predictions.append(image_result['prediction'])
            confidences.append(image_result['confidence'])
            
            # Get pattern-based prediction
            patterns = image_result['patterns']
            pattern_prediction, pattern_confidence = self._prediction_from_patterns(patterns)
            if pattern_prediction:
                predictions.append(pattern_prediction)
                confidences.append(pattern_confidence)
            
            # If OHLCV data is provided, use models for additional predictions
            if ohlcv_data is not None and not ohlcv_data.empty:
                # Calculate indicators
                indicators_df = self.indicators.calculate_all_indicators(ohlcv_data)
                
                # Use time series model
                ts_result = self.ts_models.predict_next_bar(
                    indicators_df, model_type='arima', features=['close']
                )
                
                # Convert regression prediction to classification
                if 'direction' in ts_result:
                    ts_prediction = 'BUY' if ts_result['direction'] == 'UP' else 'SELL'
                    predictions.append(ts_prediction)
                    confidences.append(ts_result.get('confidence', 0.6))
            
            # Aggregate predictions
            if not predictions:
                return {
                    'prediction': 'HOLD',
                    'confidence': 0.5,
                    'patterns': patterns,
                    'timeframe': timeframe
                }
            
            # Count occurrences of each prediction
            from collections import Counter
            pred_counter = Counter(predictions)
            
            # Get the most common prediction
            final_prediction, count = pred_counter.most_common(1)[0]
            
            # Calculate average confidence for the final prediction
            final_confidence = 0
            count = 0
            for pred, conf in zip(predictions, confidences):
                if pred == final_prediction:
                    final_confidence += conf
                    count += 1
            
            final_confidence = final_confidence / count if count > 0 else 0.5
            
            return {
                'prediction': final_prediction,
                'confidence': final_confidence,
                'patterns': patterns,
                'timeframe': timeframe
            }
        except Exception as e:
            logger.error(f"Error in ensemble prediction: {e}")
            raise
    
    def _prediction_from_patterns(self, patterns: List[str]) -> Tuple[str, float]:
        """
        Derive a prediction from detected patterns
        
        Parameters:
        -----------
        patterns : List[str]
            List of detected patterns
            
        Returns:
        --------
        Tuple[str, float]
            (prediction, confidence)
        """
        if not patterns:
            return None, 0
        
        # Count bullish and bearish patterns
        bullish_count = 0
        bearish_count = 0
        
        for pattern in patterns:
            if any(p in pattern.lower() for p in ['bullish', 'hammer', 'buy']):
                bullish_count += 1
            elif any(p in pattern.lower() for p in ['bearish', 'shooting', 'sell']):
                bearish_count += 1
        
        # Determine overall direction
        if bullish_count > bearish_count:
            return 'BUY', 0.5 + (0.1 * min(bullish_count, 5))
        elif bearish_count > bullish_count:
            return 'SELL', 0.5 + (0.1 * min(bearish_count, 5))
        
        # Neutral if equal counts
        return 'HOLD', 0.5
