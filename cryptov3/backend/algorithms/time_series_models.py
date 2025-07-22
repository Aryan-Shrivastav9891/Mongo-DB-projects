import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


class TimeSeriesModels:
    """
    Implementation of statistical and time-series models for financial data
    """

    def __init__(self):
        pass

    def check_stationarity(self, series: pd.Series) -> Dict[str, Any]:
        """
        Check if a time series is stationary using the Augmented Dickey-Fuller test
        
        Parameters:
        -----------
        series : pd.Series
            Time series data
            
        Returns:
        --------
        Dict[str, Any]
            Dictionary with test results
        """
        # Perform ADF test
        result = adfuller(series.dropna())
        
        # Format the results
        output = {
            'test_statistic': result[0],
            'p_value': result[1],
            'lags_used': result[2],
            'num_observations': result[3],
            'critical_values': result[4],
            'is_stationary': result[1] < 0.05  # If p-value < 0.05, reject null hypothesis => stationary
        }
        
        return output

    def difference_series(self, series: pd.Series, order: int = 1) -> pd.Series:
        """
        Difference a time series to make it stationary
        
        Parameters:
        -----------
        series : pd.Series
            Time series data
        order : int
            Order of differencing
            
        Returns:
        --------
        pd.Series
            Differenced series
        """
        return series.diff(order).dropna()

    def create_lag_features(self, series: pd.Series, lags: List[int]) -> pd.DataFrame:
        """
        Create lag features for time series modeling
        
        Parameters:
        -----------
        series : pd.Series
            Time series data
        lags : List[int]
            List of lag orders to create
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with lagged features
        """
        df = pd.DataFrame(series)
        column_name = series.name if series.name else 'value'
        
        for lag in lags:
            df[f'{column_name}_lag_{lag}'] = series.shift(lag)
        
        return df.dropna()

    def plot_acf_pacf(self, series: pd.Series, lags: int = 40) -> None:
        """
        Plot ACF and PACF for a time series
        
        Parameters:
        -----------
        series : pd.Series
            Time series data
        lags : int
            Number of lags to include
        """
        plt.figure(figsize=(12, 6))
        
        plt.subplot(121)
        plot_acf(series.dropna(), ax=plt.gca(), lags=lags)
        plt.title('Autocorrelation Function (ACF)')
        
        plt.subplot(122)
        plot_pacf(series.dropna(), ax=plt.gca(), lags=lags)
        plt.title('Partial Autocorrelation Function (PACF)')
        
        plt.tight_layout()
        plt.show()

    def fit_arima(self, series: pd.Series, order: Tuple[int, int, int] = (5, 1, 0)) -> Dict[str, Any]:
        """
        Fit an ARIMA model to time series data
        
        Parameters:
        -----------
        series : pd.Series
            Time series data
        order : Tuple[int, int, int]
            ARIMA model order (p, d, q)
            
        Returns:
        --------
        Dict[str, Any]
            Dictionary with model results
        """
        try:
            # Fit the model
            model = ARIMA(series, order=order)
            results = model.fit()
            
            # Return model results
            return {
                'model': results,
                'aic': results.aic,
                'bic': results.bic,
                'params': results.params.to_dict()
            }
        except Exception as e:
            return {'error': str(e)}

    def fit_sarima(self, 
                  series: pd.Series, 
                  order: Tuple[int, int, int] = (1, 1, 1),
                  seasonal_order: Tuple[int, int, int, int] = (0, 0, 0, 0)) -> Dict[str, Any]:
        """
        Fit a SARIMA model to time series data
        
        Parameters:
        -----------
        series : pd.Series
            Time series data
        order : Tuple[int, int, int]
            ARIMA model order (p, d, q)
        seasonal_order : Tuple[int, int, int, int]
            Seasonal order (P, D, Q, s)
            
        Returns:
        --------
        Dict[str, Any]
            Dictionary with model results
        """
        try:
            # Fit the model
            model = SARIMAX(series, order=order, seasonal_order=seasonal_order)
            results = model.fit()
            
            # Return model results
            return {
                'model': results,
                'aic': results.aic,
                'bic': results.bic,
                'params': results.params.to_dict()
            }
        except Exception as e:
            return {'error': str(e)}

    def fit_exponential_smoothing(self, 
                                 series: pd.Series, 
                                 trend: Optional[str] = None,
                                 seasonal: Optional[str] = None,
                                 seasonal_periods: int = None) -> Dict[str, Any]:
        """
        Fit an Exponential Smoothing model to time series data
        
        Parameters:
        -----------
        series : pd.Series
            Time series data
        trend : Optional[str]
            Type of trend component ('add', 'mul', None)
        seasonal : Optional[str]
            Type of seasonal component ('add', 'mul', None)
        seasonal_periods : int
            Number of periods in a season
            
        Returns:
        --------
        Dict[str, Any]
            Dictionary with model results
        """
        try:
            # Fit the model
            model = ExponentialSmoothing(series, 
                                        trend=trend, 
                                        seasonal=seasonal,
                                        seasonal_periods=seasonal_periods)
            results = model.fit()
            
            # Return model results
            return {
                'model': results,
                'params': {
                    'smoothing_level': results.params['smoothing_level'],
                    'smoothing_trend': results.params.get('smoothing_trend', None),
                    'smoothing_seasonal': results.params.get('smoothing_seasonal', None),
                }
            }
        except Exception as e:
            return {'error': str(e)}

    def seasonal_decomposition(self, 
                              series: pd.Series, 
                              period: int = 5, 
                              model: str = 'additive') -> Dict[str, pd.Series]:
        """
        Perform seasonal decomposition of time series
        
        Parameters:
        -----------
        series : pd.Series
            Time series data
        period : int
            Period of seasonality
        model : str
            Type of decomposition ('additive' or 'multiplicative')
            
        Returns:
        --------
        Dict[str, pd.Series]
            Dictionary with decomposition components
        """
        try:
            # Perform seasonal decomposition
            decomposition = seasonal_decompose(series, period=period, model=model)
            
            # Return components
            return {
                'trend': decomposition.trend,
                'seasonal': decomposition.seasonal,
                'residual': decomposition.resid,
                'observed': decomposition.observed
            }
        except Exception as e:
            return {'error': str(e)}

    def kalman_filter(self, series: pd.Series) -> pd.Series:
        """
        Apply Kalman filter for trend extraction
        
        Parameters:
        -----------
        series : pd.Series
            Time series data
            
        Returns:
        --------
        pd.Series
            Filtered series
        """
        # Define the state space model
        y = series.values
        
        # Simple local level model
        # y_t = mu_t + eps_t
        # mu_t = mu_{t-1} + eta_t
        
        # Initialize Kalman Filter model
        kf = sm.tsa.statespace.structural.UnobservedComponents(
            y, 'local level'
        )
        
        # Fit the model
        kf_res = kf.fit(disp=False)
        
        # Extract the filtered state
        filtered_state = kf_res.filtered_state[0]
        
        return pd.Series(filtered_state, index=series.index, name=f'{series.name}_trend')

    def find_optimal_arima_order(self, 
                               series: pd.Series,
                               max_p: int = 5,
                               max_d: int = 2,
                               max_q: int = 5) -> Tuple[Tuple[int, int, int], float]:
        """
        Find optimal ARIMA order using AIC
        
        Parameters:
        -----------
        series : pd.Series
            Time series data
        max_p : int
            Maximum value for p (AR order)
        max_d : int
            Maximum value for d (differencing order)
        max_q : int
            Maximum value for q (MA order)
            
        Returns:
        --------
        Tuple[Tuple[int, int, int], float]
            (best_order, best_aic)
        """
        best_order = None
        best_aic = float('inf')
        
        # Determine best differencing order
        for d in range(max_d + 1):
            # Difference the series if d > 0
            if d == 0:
                diff_series = series
            else:
                diff_series = self.difference_series(series, d)
            
            # Try different combinations of p and q
            for p in range(max_p + 1):
                for q in range(max_q + 1):
                    # Skip if both p and q are 0
                    if p == 0 and q == 0:
                        continue
                    
                    try:
                        # Fit ARIMA model
                        order = (p, d, q)
                        model = ARIMA(series, order=order)
                        results = model.fit()
                        
                        # Update best order if AIC is lower
                        if results.aic < best_aic:
                            best_aic = results.aic
                            best_order = order
                    except:
                        continue
        
        return best_order, best_aic

    def forecast_arima(self, 
                      series: pd.Series,
                      order: Tuple[int, int, int],
                      steps: int = 5) -> pd.Series:
        """
        Forecast future values using ARIMA
        
        Parameters:
        -----------
        series : pd.Series
            Time series data
        order : Tuple[int, int, int]
            ARIMA order (p, d, q)
        steps : int
            Number of steps to forecast
            
        Returns:
        --------
        pd.Series
            Forecasted values
        """
        # Fit the model
        model = ARIMA(series, order=order)
        results = model.fit()
        
        # Make forecast
        forecast = results.forecast(steps=steps)
        
        return forecast

    def predict_next_bar(self, 
                        df: pd.DataFrame,
                        model_type: str = 'arima',
                        features: List[str] = ['close']) -> Dict[str, Any]:
        """
        Predict the next bar's values using time series models
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with OHLCV data
        model_type : str
            Type of model to use ('arima', 'sarima', 'exp_smooth')
        features : List[str]
            List of features to predict
            
        Returns:
        --------
        Dict[str, Any]
            Predicted values for the next bar
        """
        predictions = {}
        
        for feature in features:
            if feature not in df.columns:
                continue
                
            series = df[feature]
            
            if model_type == 'arima':
                # Find optimal order
                order, _ = self.find_optimal_arima_order(series)
                
                # Make prediction
                forecast = self.forecast_arima(series, order, steps=1)
                predictions[feature] = forecast.iloc[0]
                
            elif model_type == 'sarima':
                # Simplified order selection for SARIMA
                model = SARIMAX(series, order=(1, 1, 1), seasonal_order=(0, 1, 1, 5))
                results = model.fit(disp=False)
                forecast = results.forecast(steps=1)
                predictions[feature] = forecast.iloc[0]
                
            elif model_type == 'exp_smooth':
                model = ExponentialSmoothing(series, trend='add')
                results = model.fit()
                forecast = results.forecast(steps=1)
                predictions[feature] = forecast.iloc[0]
        
        # If predicting close only, add direction and confidence
        if len(features) == 1 and 'close' in features:
            last_close = df['close'].iloc[-1]
            predicted_close = predictions['close']
            
            predictions['direction'] = 'UP' if predicted_close > last_close else 'DOWN'
            predictions['confidence'] = abs((predicted_close - last_close) / last_close)
        
        return predictions
