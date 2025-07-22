import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os
import logging
from typing import Dict, Any
from algorithms.integrated_model import IntegratedModel

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CandlestickPredictor:
    def __init__(self, model_path: str = None):
        """
        Initialize the predictor. If model_path is provided,
        uses the specified model directory. Otherwise uses 
        the default models directory or placeholder algorithms.
        """
        try:
            # Set up models directory
            models_dir = model_path if model_path and os.path.exists(model_path) else os.path.join(os.path.dirname(__file__), '..', 'models')
            
            # Initialize the integrated model
            self.integrated_model = IntegratedModel(models_dir=models_dir)
            logger.info(f"Initialized integrated model with models directory: {models_dir}")
            
            # Legacy model for compatibility
            self.model = None
            self.class_map = {0: 'BUY', 1: 'SELL', 2: 'HOLD'}
            
        except Exception as e:
            logger.error(f"Error initializing model: {e}")
            self.model = None
            self.integrated_model = None
    
    def preprocess(self, image_path: str) -> np.ndarray:
        """
        Preprocess the image for model input.
        """
        try:
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ValueError(f"Could not read image at {image_path}")
            
            img = cv2.resize(img, (256, 256))
            img = img / 255.0
            return np.expand_dims(img, axis=-1)
        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            raise
    
    def predict(self, image_path: str, timeframe: str = None, image_hash=None) -> Dict[str, Any]:
        """
        Predict candlestick pattern from image.
        
        Parameters:
        -----------
        image_path : str
            Path to the image file
        timeframe : str
            Timeframe of the chart
        image_hash : str, optional
            Hash of the image for deterministic results
        """
        try:
            print(f"Prediction service received request for {image_path} with timeframe {timeframe}")
            # Set a deterministic seed if image hash is provided
            if image_hash:
                print(f"Setting random seed from hash: {image_hash[:8]}")
                seed_value = int(image_hash[:8], 16) % (2**32 - 1)  # Ensure value fits in 32-bit integer
                print(f"Seed value: {seed_value}")
                np.random.seed(seed_value)
            
            # Verify the image file exists
            if not os.path.exists(image_path):
                print(f"ERROR: Image file does not exist: {image_path}")
                raise FileNotFoundError(f"Image file not found: {image_path}")
                
            print(f"Image file exists at {image_path}, size: {os.path.getsize(image_path)} bytes")
                
            # Use integrated model if available
            if self.integrated_model:
                print("Using integrated model for prediction")
                result = self.integrated_model.predict_from_image(image_path, timeframe, image_hash)
                
                # Add detected patterns based on prediction
                if 'patterns' not in result or not result['patterns']:
                    print("No patterns detected, adding default patterns")
                    result['patterns'] = self._get_default_patterns(result['prediction'], result['confidence'])
                
                print(f"Prediction result: {result}")
                return result
            
            # Fallback to legacy method
            print("Integrated model not available, using legacy method")
            try:
                processed = self.preprocess(image_path)
                print("Image preprocessed successfully")
            except Exception as e:
                print(f"Error preprocessing image: {str(e)}")
                raise
            
            if self.model:
                # Use actual model for prediction
                print("Using model for prediction")
                predictions = self.model.predict(np.array([processed]))
                pred_index = np.argmax(predictions)
                confidence = float(np.max(predictions))
                prediction = self.class_map[pred_index]
            else:
                # Use placeholder prediction logic based on image features
                print("Using placeholder prediction logic")
                prediction, confidence = self._placeholder_prediction(processed, timeframe)
            
            result = {
                'prediction': prediction,
                'confidence': confidence,
                'timeframe': timeframe,
                'patterns': self._get_default_patterns(prediction, confidence)
            }
            print(f"Legacy prediction result: {result}")
            return result
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            print(f"Exception in prediction service: {str(e)}")
            import traceback
            print(traceback.format_exc())
            raise
    
    def _placeholder_prediction(self, processed_img: np.ndarray, timeframe: str) -> tuple:
        """
        A placeholder prediction mechanism for demonstration when model is not available.
        Uses basic image statistics to make a "prediction".
        """
        # Calculate some basic image features
        mean_val = np.mean(processed_img)
        std_val = np.std(processed_img)
        min_val = np.min(processed_img)
        max_val = np.max(processed_img)
        
        # Simple logic based on image statistics and timeframe
        # This is just for demonstration and should be replaced with actual model inference
        
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
            if max_val - min_val > 0.7:  # High contrast
                if mean_val > 0.5:
                    return 'BUY', 0.75 + (mean_val - 0.5) / 2
                else:
                    return 'SELL', 0.80 - (mean_val / 2)
            else:  # Low contrast
                return 'HOLD', 0.65 + std_val
        
        # Default fallback
        return 'HOLD', 0.50
        
    def _get_default_patterns(self, prediction: str, confidence: float) -> list:
        """
        Generate default patterns based on prediction and confidence.
        """
        patterns = []
        
        if prediction == "BUY":
            if confidence > 0.8:
                patterns = ["Bullish Engulfing", "Morning Star"]
            elif confidence > 0.6:
                patterns = ["Hammer", "Inverted Hammer"]
            else:
                patterns = ["Potential Reversal"]
                
        elif prediction == "SELL":
            if confidence > 0.8:
                patterns = ["Bearish Engulfing", "Evening Star"]
            elif confidence > 0.6:
                patterns = ["Shooting Star", "Hanging Man"]
            else:
                patterns = ["Potential Downtrend"]
                
        elif prediction == "HOLD":
            patterns = ["Doji", "Spinning Top"]
            
        return patterns
