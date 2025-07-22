import pandas as pd
import numpy as np
from typing import Union, List, Dict, Any, Optional


class DataPreprocessor:
    """
    Handles preprocessing of OHLCV time series data for candlestick analysis
    """

    def __init__(self):
        pass

    def load_from_csv(self, file_path: str) -> pd.DataFrame:
        """Load OHLCV data from a CSV file"""
        df = pd.read_csv(file_path)
        return self._validate_and_clean_dataframe(df)

    def load_from_dict(self, data: Dict[str, List]) -> pd.DataFrame:
        """Load OHLCV data from a dictionary"""
        df = pd.DataFrame(data)
        return self._validate_and_clean_dataframe(df)

    def _validate_and_clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean the dataframe"""
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        
        # Check for required columns
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Convert timestamp to datetime if it's not already
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Set timestamp as index
        df = df.set_index('timestamp')
        
        # Handle missing values
        df = self._handle_missing_values(df)
        
        return df

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataframe"""
        # Forward fill missing values
        df = df.ffill()
        
        # For any remaining NaN values, use linear interpolation
        df = df.interpolate(method='linear')
        
        return df

    def normalize_data(self, df: pd.DataFrame, method: str = 'z-score') -> pd.DataFrame:
        """
        Normalize data using specified method
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame containing OHLCV data
        method : str
            Normalization method: 'z-score', 'min-max', 'log-returns'
            
        Returns:
        --------
        pd.DataFrame
            Normalized dataframe
        """
        if method == 'z-score':
            # Z-score normalization: (x - mean) / std
            normalized = df.copy()
            for col in ['open', 'high', 'low', 'close']:
                mean = df[col].mean()
                std = df[col].std()
                normalized[col] = (df[col] - mean) / std
            
            # Volume requires separate normalization
            normalized['volume'] = (df['volume'] - df['volume'].mean()) / df['volume'].std()
            
        elif method == 'min-max':
            # Min-max normalization: (x - min) / (max - min)
            normalized = df.copy()
            for col in df.columns:
                min_val = df[col].min()
                max_val = df[col].max()
                normalized[col] = (df[col] - min_val) / (max_val - min_val)
                
        elif method == 'log-returns':
            # Log returns: ln(P_t / P_t-1)
            normalized = df.copy()
            for col in ['open', 'high', 'low', 'close']:
                normalized[col] = np.log(df[col] / df[col].shift(1))
            
            # For volume, we use simple percentage change
            normalized['volume'] = df['volume'].pct_change()
            
            # Drop first row which will have NaN values after transformation
            normalized = normalized.iloc[1:]
            
        else:
            raise ValueError(f"Unsupported normalization method: {method}")
            
        return normalized

    def resample_data(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """
        Resample data to a different timeframe
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with timestamp index
        timeframe : str
            Target timeframe ('1min', '5min', '1h', '1d', etc.)
            
        Returns:
        --------
        pd.DataFrame
            Resampled dataframe
        """
        # Verify timestamp is the index
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame index must be a DatetimeIndex with timestamp values")
        
        # Define aggregation dictionary for OHLCV data
        agg_dict = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }
        
        # Filter for only the columns we need to aggregate
        cols_to_use = [col for col in agg_dict.keys() if col in df.columns]
        agg_dict = {col: agg_dict[col] for col in cols_to_use}
        
        # Resample the data
        resampled = df.resample(timeframe).agg(agg_dict)
        
        # Handle missing values that may have been introduced
        resampled = self._handle_missing_values(resampled)
        
        return resampled

    def compute_log_returns(self, df: pd.DataFrame, column: str = 'close') -> pd.Series:
        """Compute log returns: ln(P_t / P_t-1)"""
        return np.log(df[column] / df[column].shift(1)).dropna()

    def compute_rolling_statistics(self, 
                                  df: pd.DataFrame, 
                                  column: str = 'close', 
                                  window: int = 20) -> pd.DataFrame:
        """Compute rolling statistics (mean, std, skew, kurtosis)"""
        data = pd.DataFrame(index=df.index)
        
        # Rolling mean
        data[f'{column}_mean_{window}'] = df[column].rolling(window=window).mean()
        
        # Rolling standard deviation
        data[f'{column}_std_{window}'] = df[column].rolling(window=window).std()
        
        # Rolling skewness
        data[f'{column}_skew_{window}'] = df[column].rolling(window=window).skew()
        
        # Rolling kurtosis
        data[f'{column}_kurt_{window}'] = df[column].rolling(window=window).kurt()
        
        return data

    def compute_vwap(self, df: pd.DataFrame) -> pd.Series:
        """Compute Volume Weighted Average Price (VWAP)"""
        return (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()

    def compute_twap(self, df: pd.DataFrame, window: int = 20) -> pd.Series:
        """Compute Time Weighted Average Price (TWAP)"""
        return df['close'].rolling(window=window).mean()

    def extract_features_from_image(self, image_array: np.ndarray) -> Dict[str, float]:
        """
        Extract basic statistical features from a candlestick chart image
        
        Parameters:
        -----------
        image_array : np.ndarray
            Normalized image array
            
        Returns:
        --------
        Dict[str, float]
            Dictionary of image features
        """
        features = {}
        
        # Basic statistical features
        features['mean'] = float(np.mean(image_array))
        features['std'] = float(np.std(image_array))
        features['min'] = float(np.min(image_array))
        features['max'] = float(np.max(image_array))
        features['median'] = float(np.median(image_array))
        
        # Contrast features
        features['contrast'] = float(features['max'] - features['min'])
        
        # Gradient features (to detect edges)
        if len(image_array.shape) == 3 and image_array.shape[2] == 1:
            # If it's a 3D array with single channel
            grad_x = np.abs(np.gradient(image_array[:,:,0], axis=1))
            grad_y = np.abs(np.gradient(image_array[:,:,0], axis=0))
        else:
            # If it's a 2D array
            grad_x = np.abs(np.gradient(image_array, axis=1))
            grad_y = np.abs(np.gradient(image_array, axis=0))
            
        features['grad_mean'] = float(np.mean(grad_x + grad_y))
        features['grad_std'] = float(np.std(grad_x + grad_y))
        
        return features
