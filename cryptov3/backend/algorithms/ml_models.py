import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
import xgboost as xgb
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class MachineLearningModels:
    """
    Implementation of classical machine learning models for financial time series
    """

    def __init__(self):
        self.models = {}
        self.scalers = {}
    
    def _prepare_features(self, 
                         df: pd.DataFrame, 
                         target_col: str, 
                         feature_cols: Optional[List[str]] = None,
                         classification: bool = False,
                         train_size: float = 0.8) -> Dict[str, np.ndarray]:
        """
        Prepare features and target for ML modeling
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with features and target
        target_col : str
            Name of target column
        feature_cols : Optional[List[str]]
            List of feature columns to use
        classification : bool
            Whether this is a classification task
        train_size : float
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
        X = df[feature_cols].values
        y = df[target_col].values
        
        # For classification tasks, convert target to categorical
        if classification:
            if np.issubdtype(y.dtype, np.number) and len(np.unique(y)) < 5:
                y = y.astype(int)
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, train_size=train_size, shuffle=False
        )
        
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'feature_cols': feature_cols,
            'scaler': scaler
        }
    
    def _create_windowed_features(self, 
                                df: pd.DataFrame, 
                                window_size: int = 5,
                                target_col: str = 'close',
                                target_shift: int = 1) -> Tuple[pd.DataFrame, str]:
        """
        Create windowed features for time-series forecasting
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with time-series data
        window_size : int
            Size of the look-back window
        target_col : str
            Column to forecast
        target_shift : int
            Number of steps ahead to forecast
            
        Returns:
        --------
        Tuple[pd.DataFrame, str]
            (DataFrame with features, name of target column)
        """
        # Create a copy to avoid modifying the original
        result_df = df.copy()
        
        # Create lagged features
        for col in df.columns:
            for i in range(1, window_size + 1):
                result_df[f'{col}_lag_{i}'] = df[col].shift(i)
        
        # Create target column (future value)
        target_name = f'{target_col}_next_{target_shift}'
        result_df[target_name] = df[target_col].shift(-target_shift)
        
        # Drop rows with NaN values
        result_df = result_df.dropna()
        
        return result_df, target_name
    
    def train_random_forest(self, 
                           df: pd.DataFrame, 
                           target_col: str,
                           feature_cols: Optional[List[str]] = None,
                           classification: bool = False,
                           **kwargs) -> Dict[str, Any]:
        """
        Train a Random Forest model
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with features and target
        target_col : str
            Name of target column
        feature_cols : Optional[List[str]]
            List of feature columns to use
        classification : bool
            Whether to use a classifier or regressor
        **kwargs : 
            Additional parameters for the Random Forest model
            
        Returns:
        --------
        Dict[str, Any]
            Dictionary with model and evaluation results
        """
        # Prepare data
        data = self._prepare_features(df, target_col, feature_cols, classification)
        
        # Set default parameters
        params = {
            'n_estimators': kwargs.get('n_estimators', 100),
            'max_depth': kwargs.get('max_depth', 10),
            'random_state': kwargs.get('random_state', 42)
        }
        
        # Ensure deterministic behavior by always setting a random_state
        if 'random_state' not in params:
            params['random_state'] = 42
        
        # Create and train model
        if classification:
            model = RandomForestClassifier(**params)
            model.fit(data['X_train'], data['y_train'])
            
            # Make predictions
            y_pred = model.predict(data['X_test'])
            y_prob = model.predict_proba(data['X_test'])
            
            # Evaluate
            metrics = {
                'accuracy': accuracy_score(data['y_test'], y_pred),
                'precision': precision_score(data['y_test'], y_pred, average='weighted'),
                'recall': recall_score(data['y_test'], y_pred, average='weighted'),
                'f1': f1_score(data['y_test'], y_pred, average='weighted')
            }
            
        else:
            model = RandomForestRegressor(**params)
            model.fit(data['X_train'], data['y_train'])
            
            # Make predictions
            y_pred = model.predict(data['X_test'])
            
            # Evaluate
            metrics = {
                'mse': mean_squared_error(data['y_test'], y_pred),
                'rmse': np.sqrt(mean_squared_error(data['y_test'], y_pred)),
                'mae': mean_absolute_error(data['y_test'], y_pred),
                'r2': r2_score(data['y_test'], y_pred)
            }
        
        # Store model and scaler
        model_name = f"rf_{'clf' if classification else 'reg'}_{target_col}"
        self.models[model_name] = model
        self.scalers[model_name] = data['scaler']
        
        # Return results
        return {
            'model_name': model_name,
            'model': model,
            'metrics': metrics,
            'feature_importance': dict(zip(data['feature_cols'], model.feature_importances_))
        }
    
    def train_gradient_boosting(self, 
                              df: pd.DataFrame, 
                              target_col: str,
                              feature_cols: Optional[List[str]] = None,
                              classification: bool = False,
                              **kwargs) -> Dict[str, Any]:
        """
        Train a Gradient Boosting model
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with features and target
        target_col : str
            Name of target column
        feature_cols : Optional[List[str]]
            List of feature columns to use
        classification : bool
            Whether to use a classifier or regressor
        **kwargs : 
            Additional parameters for the Gradient Boosting model
            
        Returns:
        --------
        Dict[str, Any]
            Dictionary with model and evaluation results
        """
        # Prepare data
        data = self._prepare_features(df, target_col, feature_cols, classification)
        
        # Set default parameters
        params = {
            'n_estimators': kwargs.get('n_estimators', 100),
            'learning_rate': kwargs.get('learning_rate', 0.1),
            'max_depth': kwargs.get('max_depth', 3),
            'random_state': kwargs.get('random_state', 42)
        }
        
        # Create and train model
        if classification:
            model = GradientBoostingClassifier(**params)
            model.fit(data['X_train'], data['y_train'])
            
            # Make predictions
            y_pred = model.predict(data['X_test'])
            y_prob = model.predict_proba(data['X_test'])
            
            # Evaluate
            metrics = {
                'accuracy': accuracy_score(data['y_test'], y_pred),
                'precision': precision_score(data['y_test'], y_pred, average='weighted'),
                'recall': recall_score(data['y_test'], y_pred, average='weighted'),
                'f1': f1_score(data['y_test'], y_pred, average='weighted')
            }
            
        else:
            model = GradientBoostingRegressor(**params)
            model.fit(data['X_train'], data['y_train'])
            
            # Make predictions
            y_pred = model.predict(data['X_test'])
            
            # Evaluate
            metrics = {
                'mse': mean_squared_error(data['y_test'], y_pred),
                'rmse': np.sqrt(mean_squared_error(data['y_test'], y_pred)),
                'mae': mean_absolute_error(data['y_test'], y_pred),
                'r2': r2_score(data['y_test'], y_pred)
            }
        
        # Store model and scaler
        model_name = f"gb_{'clf' if classification else 'reg'}_{target_col}"
        self.models[model_name] = model
        self.scalers[model_name] = data['scaler']
        
        # Return results
        return {
            'model_name': model_name,
            'model': model,
            'metrics': metrics,
            'feature_importance': dict(zip(data['feature_cols'], model.feature_importances_))
        }
    
    def train_xgboost(self, 
                     df: pd.DataFrame, 
                     target_col: str,
                     feature_cols: Optional[List[str]] = None,
                     classification: bool = False,
                     **kwargs) -> Dict[str, Any]:
        """
        Train an XGBoost model
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with features and target
        target_col : str
            Name of target column
        feature_cols : Optional[List[str]]
            List of feature columns to use
        classification : bool
            Whether to use a classifier or regressor
        **kwargs : 
            Additional parameters for the XGBoost model
            
        Returns:
        --------
        Dict[str, Any]
            Dictionary with model and evaluation results
        """
        # Prepare data
        data = self._prepare_features(df, target_col, feature_cols, classification)
        
        # Set default parameters
        params = {
            'n_estimators': kwargs.get('n_estimators', 100),
            'learning_rate': kwargs.get('learning_rate', 0.1),
            'max_depth': kwargs.get('max_depth', 6),
            'random_state': kwargs.get('random_state', 42)
        }
        
        # Create and train model
        if classification:
            model = xgb.XGBClassifier(**params)
            model.fit(data['X_train'], data['y_train'])
            
            # Make predictions
            y_pred = model.predict(data['X_test'])
            y_prob = model.predict_proba(data['X_test'])
            
            # Evaluate
            metrics = {
                'accuracy': accuracy_score(data['y_test'], y_pred),
                'precision': precision_score(data['y_test'], y_pred, average='weighted'),
                'recall': recall_score(data['y_test'], y_pred, average='weighted'),
                'f1': f1_score(data['y_test'], y_pred, average='weighted')
            }
            
        else:
            model = xgb.XGBRegressor(**params)
            model.fit(data['X_train'], data['y_train'])
            
            # Make predictions
            y_pred = model.predict(data['X_test'])
            
            # Evaluate
            metrics = {
                'mse': mean_squared_error(data['y_test'], y_pred),
                'rmse': np.sqrt(mean_squared_error(data['y_test'], y_pred)),
                'mae': mean_absolute_error(data['y_test'], y_pred),
                'r2': r2_score(data['y_test'], y_pred)
            }
        
        # Store model and scaler
        model_name = f"xgb_{'clf' if classification else 'reg'}_{target_col}"
        self.models[model_name] = model
        self.scalers[model_name] = data['scaler']
        
        # Return results
        return {
            'model_name': model_name,
            'model': model,
            'metrics': metrics,
            'feature_importance': dict(zip(data['feature_cols'], model.feature_importances_))
        }
    
    def train_svm(self, 
                 df: pd.DataFrame, 
                 target_col: str,
                 feature_cols: Optional[List[str]] = None,
                 classification: bool = False,
                 **kwargs) -> Dict[str, Any]:
        """
        Train an SVM model
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with features and target
        target_col : str
            Name of target column
        feature_cols : Optional[List[str]]
            List of feature columns to use
        classification : bool
            Whether to use a classifier or regressor
        **kwargs : 
            Additional parameters for the SVM model
            
        Returns:
        --------
        Dict[str, Any]
            Dictionary with model and evaluation results
        """
        # Prepare data
        data = self._prepare_features(df, target_col, feature_cols, classification)
        
        # Set default parameters
        params = {
            'C': kwargs.get('C', 1.0),
            'kernel': kwargs.get('kernel', 'rbf'),
            'gamma': kwargs.get('gamma', 'scale'),
            'probability': True if classification else kwargs.get('probability', False)
        }
        
        # Create and train model
        if classification:
            model = SVC(**params)
            model.fit(data['X_train'], data['y_train'])
            
            # Make predictions
            y_pred = model.predict(data['X_test'])
            y_prob = model.predict_proba(data['X_test']) if params['probability'] else None
            
            # Evaluate
            metrics = {
                'accuracy': accuracy_score(data['y_test'], y_pred),
                'precision': precision_score(data['y_test'], y_pred, average='weighted'),
                'recall': recall_score(data['y_test'], y_pred, average='weighted'),
                'f1': f1_score(data['y_test'], y_pred, average='weighted')
            }
            
        else:
            model = SVR(**params)
            model.fit(data['X_train'], data['y_train'])
            
            # Make predictions
            y_pred = model.predict(data['X_test'])
            
            # Evaluate
            metrics = {
                'mse': mean_squared_error(data['y_test'], y_pred),
                'rmse': np.sqrt(mean_squared_error(data['y_test'], y_pred)),
                'mae': mean_absolute_error(data['y_test'], y_pred),
                'r2': r2_score(data['y_test'], y_pred)
            }
        
        # Store model and scaler
        model_name = f"svm_{'clf' if classification else 'reg'}_{target_col}"
        self.models[model_name] = model
        self.scalers[model_name] = data['scaler']
        
        # Return results
        return {
            'model_name': model_name,
            'model': model,
            'metrics': metrics
        }
    
    def train_knn(self, 
                 df: pd.DataFrame, 
                 target_col: str,
                 feature_cols: Optional[List[str]] = None,
                 classification: bool = False,
                 **kwargs) -> Dict[str, Any]:
        """
        Train a k-Nearest Neighbors model
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with features and target
        target_col : str
            Name of target column
        feature_cols : Optional[List[str]]
            List of feature columns to use
        classification : bool
            Whether to use a classifier or regressor
        **kwargs : 
            Additional parameters for the kNN model
            
        Returns:
        --------
        Dict[str, Any]
            Dictionary with model and evaluation results
        """
        # Prepare data
        data = self._prepare_features(df, target_col, feature_cols, classification)
        
        # Set default parameters
        params = {
            'n_neighbors': kwargs.get('n_neighbors', 5),
            'weights': kwargs.get('weights', 'uniform'),
            'p': kwargs.get('p', 2)  # Euclidean distance by default
        }
        
        # Create and train model
        if classification:
            model = KNeighborsClassifier(**params)
            model.fit(data['X_train'], data['y_train'])
            
            # Make predictions
            y_pred = model.predict(data['X_test'])
            y_prob = model.predict_proba(data['X_test'])
            
            # Evaluate
            metrics = {
                'accuracy': accuracy_score(data['y_test'], y_pred),
                'precision': precision_score(data['y_test'], y_pred, average='weighted'),
                'recall': recall_score(data['y_test'], y_pred, average='weighted'),
                'f1': f1_score(data['y_test'], y_pred, average='weighted')
            }
            
        else:
            model = KNeighborsRegressor(**params)
            model.fit(data['X_train'], data['y_train'])
            
            # Make predictions
            y_pred = model.predict(data['X_test'])
            
            # Evaluate
            metrics = {
                'mse': mean_squared_error(data['y_test'], y_pred),
                'rmse': np.sqrt(mean_squared_error(data['y_test'], y_pred)),
                'mae': mean_absolute_error(data['y_test'], y_pred),
                'r2': r2_score(data['y_test'], y_pred)
            }
        
        # Store model and scaler
        model_name = f"knn_{'clf' if classification else 'reg'}_{target_col}"
        self.models[model_name] = model
        self.scalers[model_name] = data['scaler']
        
        # Return results
        return {
            'model_name': model_name,
            'model': model,
            'metrics': metrics
        }
    
    def predict_with_model(self, 
                         model_name: str, 
                         features: pd.DataFrame) -> Dict[str, Any]:
        """
        Make predictions using a trained model
        
        Parameters:
        -----------
        model_name : str
            Name of the model to use
        features : pd.DataFrame
            DataFrame with features for prediction
            
        Returns:
        --------
        Dict[str, Any]
            Prediction results
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found")
        
        # Get model and scaler
        model = self.models[model_name]
        scaler = self.scalers.get(model_name)
        
        # Scale features if a scaler is available
        if scaler:
            features_scaled = scaler.transform(features)
        else:
            features_scaled = features
        
        # Make predictions
        if hasattr(model, 'predict_proba'):
            # Classification model with probability
            y_pred = model.predict(features_scaled)
            y_prob = model.predict_proba(features_scaled)
            
            # Get the class with highest probability
            best_class_idx = np.argmax(y_prob, axis=1)[0]
            confidence = y_prob[0, best_class_idx]
            
            return {
                'prediction': y_pred[0],
                'confidence': float(confidence),
                'probabilities': {i: float(p) for i, p in enumerate(y_prob[0])}
            }
            
        else:
            # Regression model
            y_pred = model.predict(features_scaled)
            
            return {
                'prediction': float(y_pred[0])
            }
    
    def train_direction_model(self, 
                             df: pd.DataFrame, 
                             window_size: int = 10,
                             model_type: str = 'xgboost') -> Dict[str, Any]:
        """
        Train a model to predict price direction (up/down)
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with OHLCV data
        window_size : int
            Size of the look-back window
        model_type : str
            Type of model to use ('random_forest', 'xgboost', 'svm', 'knn')
            
        Returns:
        --------
        Dict[str, Any]
            Model training results
        """
        # Create features with technical indicators
        from .technical_indicators import TechnicalIndicators
        ti = TechnicalIndicators()
        df_with_indicators = ti.calculate_all_indicators(df)
        
        # Create direction target (1 for up, 0 for down)
        df_with_indicators['direction'] = (df_with_indicators['close'].shift(-1) > df_with_indicators['close']).astype(int)
        
        # Create lagged features
        feature_df, target_name = self._create_windowed_features(
            df_with_indicators, window_size=window_size, target_col='direction', target_shift=0
        )
        
        # Select features (excluding the target and original OHLCV)
        feature_cols = [col for col in feature_df.columns 
                       if col != target_name and not any(x in col for x in ['open', 'high', 'low', 'close', 'volume'])]
        
        # Train the specified model
        if model_type == 'random_forest':
            result = self.train_random_forest(feature_df, target_name, feature_cols, classification=True)
        elif model_type == 'xgboost':
            result = self.train_xgboost(feature_df, target_name, feature_cols, classification=True)
        elif model_type == 'svm':
            result = self.train_svm(feature_df, target_name, feature_cols, classification=True)
        elif model_type == 'knn':
            result = self.train_knn(feature_df, target_name, feature_cols, classification=True)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        return result
