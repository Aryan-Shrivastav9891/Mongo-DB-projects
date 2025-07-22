import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, GRU, Conv1D, MaxPooling1D
from tensorflow.keras.layers import Dropout, Flatten, Input, Concatenate, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class DeepLearningModels:
    """
    Implementation of deep learning models for time series prediction
    """
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        
    def _prepare_sequence_data(self, 
                              df: pd.DataFrame,
                              target_col: str = 'close',
                              feature_cols: Optional[List[str]] = None,
                              sequence_length: int = 10,
                              forecast_horizon: int = 1,
                              train_split: float = 0.8) -> Dict[str, np.ndarray]:
        """
        Prepare sequence data for deep learning models
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with time series data
        target_col : str
            Name of target column to predict
        feature_cols : Optional[List[str]]
            List of feature columns to use as inputs
        sequence_length : int
            Length of input sequences
        forecast_horizon : int
            Number of steps ahead to forecast
        train_split : float
            Fraction of data to use for training
            
        Returns:
        --------
        Dict[str, np.ndarray]
            Dictionary with prepared data
        """
        # Use all columns except target if feature_cols not specified
        if feature_cols is None:
            feature_cols = [col for col in df.columns if col != target_col]
            
        # Extract features and target
        features = df[feature_cols].values
        target = df[target_col].values
        
        # Scale features
        feature_scaler = MinMaxScaler(feature_range=(0, 1))
        features_scaled = feature_scaler.fit_transform(features)
        
        # Scale target
        target_scaler = MinMaxScaler(feature_range=(0, 1))
        target_scaled = target_scaler.fit_transform(target.reshape(-1, 1))
        
        # Create sequences
        X, y = [], []
        for i in range(len(features_scaled) - sequence_length - forecast_horizon + 1):
            X.append(features_scaled[i:(i + sequence_length)])
            y.append(target_scaled[i + sequence_length + forecast_horizon - 1])
        
        X = np.array(X)
        y = np.array(y)
        
        # Split into train and test sets
        train_size = int(len(X) * train_split)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'feature_scaler': feature_scaler,
            'target_scaler': target_scaler,
            'feature_cols': feature_cols
        }
    
    def build_lstm_model(self, 
                        input_shape: Tuple[int, int],
                        units: List[int] = [64, 32],
                        dropout_rate: float = 0.2) -> tf.keras.Model:
        """
        Build a LSTM model for time series prediction
        
        Parameters:
        -----------
        input_shape : Tuple[int, int]
            Shape of input data (sequence_length, n_features)
        units : List[int]
            List of units for each LSTM layer
        dropout_rate : float
            Dropout rate for regularization
            
        Returns:
        --------
        tf.keras.Model
            Compiled LSTM model
        """
        model = Sequential()
        
        # First LSTM layer with return_sequences=True if there are more LSTM layers
        model.add(LSTM(units[0], activation='relu', return_sequences=len(units) > 1,
                      input_shape=input_shape))
        model.add(Dropout(dropout_rate))
        
        # Additional LSTM layers
        for i in range(1, len(units)):
            return_sequences = i < len(units) - 1
            model.add(LSTM(units[i], activation='relu', return_sequences=return_sequences))
            model.add(Dropout(dropout_rate))
        
        # Output layer
        model.add(Dense(1))
        
        # Compile the model
        model.compile(optimizer=Adam(), loss='mean_squared_error')
        
        return model
    
    def build_gru_model(self, 
                       input_shape: Tuple[int, int],
                       units: List[int] = [64, 32],
                       dropout_rate: float = 0.2) -> tf.keras.Model:
        """
        Build a GRU model for time series prediction
        
        Parameters:
        -----------
        input_shape : Tuple[int, int]
            Shape of input data (sequence_length, n_features)
        units : List[int]
            List of units for each GRU layer
        dropout_rate : float
            Dropout rate for regularization
            
        Returns:
        --------
        tf.keras.Model
            Compiled GRU model
        """
        model = Sequential()
        
        # First GRU layer with return_sequences=True if there are more GRU layers
        model.add(GRU(units[0], activation='relu', return_sequences=len(units) > 1,
                     input_shape=input_shape))
        model.add(Dropout(dropout_rate))
        
        # Additional GRU layers
        for i in range(1, len(units)):
            return_sequences = i < len(units) - 1
            model.add(GRU(units[i], activation='relu', return_sequences=return_sequences))
            model.add(Dropout(dropout_rate))
        
        # Output layer
        model.add(Dense(1))
        
        # Compile the model
        model.compile(optimizer=Adam(), loss='mean_squared_error')
        
        return model
    
    def build_cnn_model(self, 
                       input_shape: Tuple[int, int],
                       filters: List[int] = [32, 64],
                       kernel_size: int = 3,
                       pool_size: int = 2,
                       dense_units: List[int] = [64]) -> tf.keras.Model:
        """
        Build a 1D CNN model for time series prediction
        
        Parameters:
        -----------
        input_shape : Tuple[int, int]
            Shape of input data (sequence_length, n_features)
        filters : List[int]
            List of filters for each Conv1D layer
        kernel_size : int
            Size of the convolution kernel
        pool_size : int
            Size of the max pooling window
        dense_units : List[int]
            List of units for dense layers
            
        Returns:
        --------
        tf.keras.Model
            Compiled CNN model
        """
        model = Sequential()
        
        # First Conv1D layer
        model.add(Conv1D(filters[0], kernel_size=kernel_size, activation='relu',
                         input_shape=input_shape))
        model.add(MaxPooling1D(pool_size=pool_size))
        
        # Additional Conv1D layers
        for filter_size in filters[1:]:
            model.add(Conv1D(filter_size, kernel_size=kernel_size, activation='relu'))
            model.add(MaxPooling1D(pool_size=pool_size))
        
        # Flatten layer
        model.add(Flatten())
        
        # Dense layers
        for units in dense_units:
            model.add(Dense(units, activation='relu'))
        
        # Output layer
        model.add(Dense(1))
        
        # Compile the model
        model.compile(optimizer=Adam(), loss='mean_squared_error')
        
        return model
    
    def build_cnn_lstm_model(self, 
                            input_shape: Tuple[int, int],
                            cnn_filters: List[int] = [32, 64],
                            kernel_size: int = 3,
                            pool_size: int = 2,
                            lstm_units: List[int] = [50],
                            dropout_rate: float = 0.2) -> tf.keras.Model:
        """
        Build a hybrid CNN-LSTM model for time series prediction
        
        Parameters:
        -----------
        input_shape : Tuple[int, int]
            Shape of input data (sequence_length, n_features)
        cnn_filters : List[int]
            List of filters for each Conv1D layer
        kernel_size : int
            Size of the convolution kernel
        pool_size : int
            Size of the max pooling window
        lstm_units : List[int]
            List of units for LSTM layers
        dropout_rate : float
            Dropout rate for regularization
            
        Returns:
        --------
        tf.keras.Model
            Compiled CNN-LSTM model
        """
        model = Sequential()
        
        # CNN layers for feature extraction
        model.add(Conv1D(cnn_filters[0], kernel_size=kernel_size, activation='relu',
                         input_shape=input_shape))
        model.add(MaxPooling1D(pool_size=pool_size))
        
        # Additional CNN layers
        for filter_size in cnn_filters[1:]:
            model.add(Conv1D(filter_size, kernel_size=kernel_size, activation='relu'))
            model.add(MaxPooling1D(pool_size=pool_size))
        
        # LSTM layers
        for i, units in enumerate(lstm_units):
            return_sequences = i < len(lstm_units) - 1
            model.add(LSTM(units, activation='relu', return_sequences=return_sequences))
            model.add(Dropout(dropout_rate))
        
        # Output layer
        model.add(Dense(1))
        
        # Compile the model
        model.compile(optimizer=Adam(), loss='mean_squared_error')
        
        return model
    
    def train_lstm_model(self, 
                        df: pd.DataFrame,
                        target_col: str = 'close',
                        feature_cols: Optional[List[str]] = None,
                        sequence_length: int = 10,
                        forecast_horizon: int = 1,
                        units: List[int] = [64, 32],
                        dropout_rate: float = 0.2,
                        epochs: int = 50,
                        batch_size: int = 32) -> Dict[str, Any]:
        """
        Train a LSTM model for time series prediction
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with time series data
        target_col : str
            Name of target column to predict
        feature_cols : Optional[List[str]]
            List of feature columns to use as inputs
        sequence_length : int
            Length of input sequences
        forecast_horizon : int
            Number of steps ahead to forecast
        units : List[int]
            List of units for each LSTM layer
        dropout_rate : float
            Dropout rate for regularization
        epochs : int
            Number of training epochs
        batch_size : int
            Batch size for training
            
        Returns:
        --------
        Dict[str, Any]
            Dictionary with model and training history
        """
        # Prepare sequence data
        data = self._prepare_sequence_data(
            df, target_col, feature_cols, sequence_length, forecast_horizon
        )
        
        # Build the model
        input_shape = (data['X_train'].shape[1], data['X_train'].shape[2])
        model = self.build_lstm_model(input_shape, units, dropout_rate)
        
        # Early stopping callback
        early_stopping = EarlyStopping(
            monitor='val_loss', patience=10, restore_best_weights=True
        )
        
        # Train the model
        history = model.fit(
            data['X_train'], data['y_train'],
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(data['X_test'], data['y_test']),
            callbacks=[early_stopping],
            verbose=0
        )
        
        # Evaluate the model
        train_loss = model.evaluate(data['X_train'], data['y_train'], verbose=0)
        test_loss = model.evaluate(data['X_test'], data['y_test'], verbose=0)
        
        # Make predictions
        y_pred = model.predict(data['X_test'])
        
        # Inverse transform predictions and actual values
        y_pred_inv = data['target_scaler'].inverse_transform(y_pred)
        y_test_inv = data['target_scaler'].inverse_transform(data['y_test'])
        
        # Calculate metrics
        mse = np.mean((y_pred_inv - y_test_inv) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_pred_inv - y_test_inv))
        
        # Store model and scalers
        model_name = f"lstm_{target_col}_{sequence_length}_{forecast_horizon}"
        self.models[model_name] = model
        self.scalers[model_name] = {
            'feature_scaler': data['feature_scaler'],
            'target_scaler': data['target_scaler'],
            'feature_cols': data['feature_cols'],
            'sequence_length': sequence_length
        }
        
        # Return results
        return {
            'model_name': model_name,
            'model': model,
            'history': history.history,
            'metrics': {
                'train_loss': train_loss,
                'test_loss': test_loss,
                'mse': mse,
                'rmse': rmse,
                'mae': mae
            }
        }
    
    def train_gru_model(self, 
                       df: pd.DataFrame,
                       target_col: str = 'close',
                       feature_cols: Optional[List[str]] = None,
                       sequence_length: int = 10,
                       forecast_horizon: int = 1,
                       units: List[int] = [64, 32],
                       dropout_rate: float = 0.2,
                       epochs: int = 50,
                       batch_size: int = 32) -> Dict[str, Any]:
        """
        Train a GRU model for time series prediction
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with time series data
        target_col : str
            Name of target column to predict
        feature_cols : Optional[List[str]]
            List of feature columns to use as inputs
        sequence_length : int
            Length of input sequences
        forecast_horizon : int
            Number of steps ahead to forecast
        units : List[int]
            List of units for each GRU layer
        dropout_rate : float
            Dropout rate for regularization
        epochs : int
            Number of training epochs
        batch_size : int
            Batch size for training
            
        Returns:
        --------
        Dict[str, Any]
            Dictionary with model and training history
        """
        # Prepare sequence data
        data = self._prepare_sequence_data(
            df, target_col, feature_cols, sequence_length, forecast_horizon
        )
        
        # Build the model
        input_shape = (data['X_train'].shape[1], data['X_train'].shape[2])
        model = self.build_gru_model(input_shape, units, dropout_rate)
        
        # Early stopping callback
        early_stopping = EarlyStopping(
            monitor='val_loss', patience=10, restore_best_weights=True
        )
        
        # Train the model
        history = model.fit(
            data['X_train'], data['y_train'],
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(data['X_test'], data['y_test']),
            callbacks=[early_stopping],
            verbose=0
        )
        
        # Evaluate the model
        train_loss = model.evaluate(data['X_train'], data['y_train'], verbose=0)
        test_loss = model.evaluate(data['X_test'], data['y_test'], verbose=0)
        
        # Make predictions
        y_pred = model.predict(data['X_test'])
        
        # Inverse transform predictions and actual values
        y_pred_inv = data['target_scaler'].inverse_transform(y_pred)
        y_test_inv = data['target_scaler'].inverse_transform(data['y_test'])
        
        # Calculate metrics
        mse = np.mean((y_pred_inv - y_test_inv) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_pred_inv - y_test_inv))
        
        # Store model and scalers
        model_name = f"gru_{target_col}_{sequence_length}_{forecast_horizon}"
        self.models[model_name] = model
        self.scalers[model_name] = {
            'feature_scaler': data['feature_scaler'],
            'target_scaler': data['target_scaler'],
            'feature_cols': data['feature_cols'],
            'sequence_length': sequence_length
        }
        
        # Return results
        return {
            'model_name': model_name,
            'model': model,
            'history': history.history,
            'metrics': {
                'train_loss': train_loss,
                'test_loss': test_loss,
                'mse': mse,
                'rmse': rmse,
                'mae': mae
            }
        }
    
    def train_cnn_model(self, 
                       df: pd.DataFrame,
                       target_col: str = 'close',
                       feature_cols: Optional[List[str]] = None,
                       sequence_length: int = 10,
                       forecast_horizon: int = 1,
                       filters: List[int] = [32, 64],
                       kernel_size: int = 3,
                       pool_size: int = 2,
                       dense_units: List[int] = [64],
                       epochs: int = 50,
                       batch_size: int = 32) -> Dict[str, Any]:
        """
        Train a 1D CNN model for time series prediction
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with time series data
        target_col : str
            Name of target column to predict
        feature_cols : Optional[List[str]]
            List of feature columns to use as inputs
        sequence_length : int
            Length of input sequences
        forecast_horizon : int
            Number of steps ahead to forecast
        filters : List[int]
            List of filters for each Conv1D layer
        kernel_size : int
            Size of the convolution kernel
        pool_size : int
            Size of the max pooling window
        dense_units : List[int]
            List of units for dense layers
        epochs : int
            Number of training epochs
        batch_size : int
            Batch size for training
            
        Returns:
        --------
        Dict[str, Any]
            Dictionary with model and training history
        """
        # Prepare sequence data
        data = self._prepare_sequence_data(
            df, target_col, feature_cols, sequence_length, forecast_horizon
        )
        
        # Build the model
        input_shape = (data['X_train'].shape[1], data['X_train'].shape[2])
        model = self.build_cnn_model(input_shape, filters, kernel_size, pool_size, dense_units)
        
        # Early stopping callback
        early_stopping = EarlyStopping(
            monitor='val_loss', patience=10, restore_best_weights=True
        )
        
        # Train the model
        history = model.fit(
            data['X_train'], data['y_train'],
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(data['X_test'], data['y_test']),
            callbacks=[early_stopping],
            verbose=0
        )
        
        # Evaluate the model
        train_loss = model.evaluate(data['X_train'], data['y_train'], verbose=0)
        test_loss = model.evaluate(data['X_test'], data['y_test'], verbose=0)
        
        # Make predictions
        y_pred = model.predict(data['X_test'])
        
        # Inverse transform predictions and actual values
        y_pred_inv = data['target_scaler'].inverse_transform(y_pred)
        y_test_inv = data['target_scaler'].inverse_transform(data['y_test'])
        
        # Calculate metrics
        mse = np.mean((y_pred_inv - y_test_inv) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_pred_inv - y_test_inv))
        
        # Store model and scalers
        model_name = f"cnn_{target_col}_{sequence_length}_{forecast_horizon}"
        self.models[model_name] = model
        self.scalers[model_name] = {
            'feature_scaler': data['feature_scaler'],
            'target_scaler': data['target_scaler'],
            'feature_cols': data['feature_cols'],
            'sequence_length': sequence_length
        }
        
        # Return results
        return {
            'model_name': model_name,
            'model': model,
            'history': history.history,
            'metrics': {
                'train_loss': train_loss,
                'test_loss': test_loss,
                'mse': mse,
                'rmse': rmse,
                'mae': mae
            }
        }
    
    def train_cnn_lstm_model(self, 
                            df: pd.DataFrame,
                            target_col: str = 'close',
                            feature_cols: Optional[List[str]] = None,
                            sequence_length: int = 10,
                            forecast_horizon: int = 1,
                            cnn_filters: List[int] = [32, 64],
                            kernel_size: int = 3,
                            pool_size: int = 2,
                            lstm_units: List[int] = [50],
                            dropout_rate: float = 0.2,
                            epochs: int = 50,
                            batch_size: int = 32) -> Dict[str, Any]:
        """
        Train a hybrid CNN-LSTM model for time series prediction
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with time series data
        target_col : str
            Name of target column to predict
        feature_cols : Optional[List[str]]
            List of feature columns to use as inputs
        sequence_length : int
            Length of input sequences
        forecast_horizon : int
            Number of steps ahead to forecast
        cnn_filters : List[int]
            List of filters for each Conv1D layer
        kernel_size : int
            Size of the convolution kernel
        pool_size : int
            Size of the max pooling window
        lstm_units : List[int]
            List of units for LSTM layers
        dropout_rate : float
            Dropout rate for regularization
        epochs : int
            Number of training epochs
        batch_size : int
            Batch size for training
            
        Returns:
        --------
        Dict[str, Any]
            Dictionary with model and training history
        """
        # Prepare sequence data
        data = self._prepare_sequence_data(
            df, target_col, feature_cols, sequence_length, forecast_horizon
        )
        
        # Build the model
        input_shape = (data['X_train'].shape[1], data['X_train'].shape[2])
        model = self.build_cnn_lstm_model(
            input_shape, cnn_filters, kernel_size, pool_size, lstm_units, dropout_rate
        )
        
        # Early stopping callback
        early_stopping = EarlyStopping(
            monitor='val_loss', patience=10, restore_best_weights=True
        )
        
        # Train the model
        history = model.fit(
            data['X_train'], data['y_train'],
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(data['X_test'], data['y_test']),
            callbacks=[early_stopping],
            verbose=0
        )
        
        # Evaluate the model
        train_loss = model.evaluate(data['X_train'], data['y_train'], verbose=0)
        test_loss = model.evaluate(data['X_test'], data['y_test'], verbose=0)
        
        # Make predictions
        y_pred = model.predict(data['X_test'])
        
        # Inverse transform predictions and actual values
        y_pred_inv = data['target_scaler'].inverse_transform(y_pred)
        y_test_inv = data['target_scaler'].inverse_transform(data['y_test'])
        
        # Calculate metrics
        mse = np.mean((y_pred_inv - y_test_inv) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_pred_inv - y_test_inv))
        
        # Store model and scalers
        model_name = f"cnn_lstm_{target_col}_{sequence_length}_{forecast_horizon}"
        self.models[model_name] = model
        self.scalers[model_name] = {
            'feature_scaler': data['feature_scaler'],
            'target_scaler': data['target_scaler'],
            'feature_cols': data['feature_cols'],
            'sequence_length': sequence_length
        }
        
        # Return results
        return {
            'model_name': model_name,
            'model': model,
            'history': history.history,
            'metrics': {
                'train_loss': train_loss,
                'test_loss': test_loss,
                'mse': mse,
                'rmse': rmse,
                'mae': mae
            }
        }
    
    def predict_with_dl_model(self, 
                            model_name: str, 
                            df: pd.DataFrame) -> Dict[str, Any]:
        """
        Make predictions using a trained deep learning model
        
        Parameters:
        -----------
        model_name : str
            Name of the model to use
        df : pd.DataFrame
            DataFrame with features for prediction
            
        Returns:
        --------
        Dict[str, Any]
            Prediction results
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found")
        
        # Get model and scalers
        model = self.models[model_name]
        model_info = self.scalers[model_name]
        
        # Extract required data
        feature_scaler = model_info['feature_scaler']
        target_scaler = model_info['target_scaler']
        feature_cols = model_info['feature_cols']
        sequence_length = model_info['sequence_length']
        
        # Check if we have enough data
        if len(df) < sequence_length:
            raise ValueError(f"Insufficient data: need at least {sequence_length} rows")
        
        # Extract features
        features = df[feature_cols].values[-sequence_length:]
        
        # Scale features
        features_scaled = feature_scaler.transform(features)
        
        # Reshape for model input
        features_reshaped = np.expand_dims(features_scaled, axis=0)
        
        # Make prediction
        prediction_scaled = model.predict(features_reshaped)
        
        # Inverse transform prediction
        prediction = target_scaler.inverse_transform(prediction_scaled)[0][0]
        
        # Calculate confidence (simplified approach)
        confidence = 0.7  # Default confidence
        
        # Get last known value for direction
        last_value = df[feature_cols[0]].values[-1]
        
        # Determine direction
        direction = 'UP' if prediction > last_value else 'DOWN'
        
        return {
            'prediction': float(prediction),
            'confidence': confidence,
            'direction': direction
        }
        
    def save_model(self, model_name: str, directory: str) -> str:
        """
        Save a trained model to disk
        
        Parameters:
        -----------
        model_name : str
            Name of the model to save
        directory : str
            Directory to save the model in
            
        Returns:
        --------
        str
            Path to the saved model
        """
        import os
        import pickle
        
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found")
        
        # Create directory if it doesn't exist
        os.makedirs(directory, exist_ok=True)
        
        # Get model
        model = self.models[model_name]
        
        # Save model
        model_path = os.path.join(directory, f"{model_name}.h5")
        model.save(model_path)
        
        # Save scalers and other info
        info_path = os.path.join(directory, f"{model_name}_info.pkl")
        with open(info_path, 'wb') as f:
            pickle.dump(self.scalers[model_name], f)
        
        return model_path
    
    def load_model(self, model_name: str, directory: str) -> None:
        """
        Load a trained model from disk
        
        Parameters:
        -----------
        model_name : str
            Name of the model to load
        directory : str
            Directory where the model is saved
        """
        import os
        import pickle
        from tensorflow.keras.models import load_model
        
        # Check if model files exist
        model_path = os.path.join(directory, f"{model_name}.h5")
        info_path = os.path.join(directory, f"{model_name}_info.pkl")
        
        if not os.path.exists(model_path) or not os.path.exists(info_path):
            raise FileNotFoundError(f"Model files for '{model_name}' not found in {directory}")
        
        # Load model
        model = load_model(model_path)
        
        # Load scalers and other info
        with open(info_path, 'rb') as f:
            model_info = pickle.load(f)
        
        # Store in instance variables
        self.models[model_name] = model
        self.scalers[model_name] = model_info
