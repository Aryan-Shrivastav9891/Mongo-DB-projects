import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union


class TechnicalIndicators:
    """
    Implementation of traditional technical indicators for OHLCV data
    """

    def __init__(self):
        pass

    def simple_moving_average(self, data: pd.Series, window: int = 20) -> pd.Series:
        """
        Calculate Simple Moving Average (SMA)
        
        Parameters:
        -----------
        data : pd.Series
            Price series (typically close prices)
        window : int
            Lookback window in periods
            
        Returns:
        --------
        pd.Series
            Simple Moving Average values
        """
        return data.rolling(window=window).mean()

    def exponential_moving_average(self, data: pd.Series, span: int = 20) -> pd.Series:
        """
        Calculate Exponential Moving Average (EMA)
        
        Parameters:
        -----------
        data : pd.Series
            Price series (typically close prices)
        span : int
            Span for EMA calculation
            
        Returns:
        --------
        pd.Series
            Exponential Moving Average values
        """
        return data.ewm(span=span, adjust=False).mean()

    def double_exponential_moving_average(self, data: pd.Series, span: int = 20) -> pd.Series:
        """
        Calculate Double Exponential Moving Average (DEMA)
        DEMA = 2 * EMA - EMA(EMA)
        
        Parameters:
        -----------
        data : pd.Series
            Price series (typically close prices)
        span : int
            Span for EMA calculation
            
        Returns:
        --------
        pd.Series
            DEMA values
        """
        ema = self.exponential_moving_average(data, span)
        ema_of_ema = self.exponential_moving_average(ema, span)
        return 2 * ema - ema_of_ema

    def triple_exponential_moving_average(self, data: pd.Series, span: int = 20) -> pd.Series:
        """
        Calculate Triple Exponential Moving Average (TEMA)
        TEMA = 3 * EMA - 3 * EMA(EMA) + EMA(EMA(EMA))
        
        Parameters:
        -----------
        data : pd.Series
            Price series (typically close prices)
        span : int
            Span for EMA calculation
            
        Returns:
        --------
        pd.Series
            TEMA values
        """
        ema = self.exponential_moving_average(data, span)
        ema_of_ema = self.exponential_moving_average(ema, span)
        ema_of_ema_of_ema = self.exponential_moving_average(ema_of_ema, span)
        return 3 * ema - 3 * ema_of_ema + ema_of_ema_of_ema

    def relative_strength_index(self, data: pd.Series, window: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index (RSI)
        RSI = 100 - (100 / (1 + RS))
        where RS = Average Gain / Average Loss
        
        Parameters:
        -----------
        data : pd.Series
            Price series (typically close prices)
        window : int
            Lookback window in periods
            
        Returns:
        --------
        pd.Series
            RSI values (0-100)
        """
        delta = data.diff()
        
        # Make two series: one for gains and one for losses
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        
        # Calculate average gain and loss
        avg_gain = gain.rolling(window=window).mean()
        avg_loss = loss.rolling(window=window).mean()
        
        # Calculate RS
        rs = avg_gain / avg_loss
        
        # Calculate RSI
        rsi = 100 - (100 / (1 + rs))
        
        return rsi

    def stochastic_oscillator(self, 
                             high: pd.Series, 
                             low: pd.Series, 
                             close: pd.Series, 
                             k_window: int = 14,
                             d_window: int = 3) -> Dict[str, pd.Series]:
        """
        Calculate Stochastic Oscillator
        %K = (Current Close - Lowest Low) / (Highest High - Lowest Low) * 100
        %D = 3-day SMA of %K
        
        Parameters:
        -----------
        high : pd.Series
            High prices
        low : pd.Series
            Low prices
        close : pd.Series
            Close prices
        k_window : int
            Lookback window for %K
        d_window : int
            Smoothing window for %D
            
        Returns:
        --------
        Dict[str, pd.Series]
            Dictionary with %K and %D
        """
        # Calculate %K
        lowest_low = low.rolling(window=k_window).min()
        highest_high = high.rolling(window=k_window).max()
        
        k = ((close - lowest_low) / (highest_high - lowest_low)) * 100
        
        # Calculate %D
        d = k.rolling(window=d_window).mean()
        
        return {'K': k, 'D': d}

    def rate_of_change(self, data: pd.Series, window: int = 14) -> pd.Series:
        """
        Calculate Rate of Change (ROC)
        ROC = ((Current Price / Price n periods ago) - 1) * 100
        
        Parameters:
        -----------
        data : pd.Series
            Price series (typically close prices)
        window : int
            Lookback window in periods
            
        Returns:
        --------
        pd.Series
            ROC values
        """
        return ((data / data.shift(window)) - 1) * 100

    def macd(self, 
            data: pd.Series, 
            fast_span: int = 12, 
            slow_span: int = 26, 
            signal_span: int = 9) -> Dict[str, pd.Series]:
        """
        Calculate Moving Average Convergence Divergence (MACD)
        MACD Line = Fast EMA - Slow EMA
        Signal Line = EMA of MACD Line
        Histogram = MACD Line - Signal Line
        
        Parameters:
        -----------
        data : pd.Series
            Price series (typically close prices)
        fast_span : int
            Span for fast EMA
        slow_span : int
            Span for slow EMA
        signal_span : int
            Span for signal line EMA
            
        Returns:
        --------
        Dict[str, pd.Series]
            Dictionary with MACD line, signal line, and histogram
        """
        # Calculate fast and slow EMAs
        fast_ema = self.exponential_moving_average(data, span=fast_span)
        slow_ema = self.exponential_moving_average(data, span=slow_span)
        
        # Calculate MACD line
        macd_line = fast_ema - slow_ema
        
        # Calculate signal line
        signal_line = self.exponential_moving_average(macd_line, span=signal_span)
        
        # Calculate histogram
        histogram = macd_line - signal_line
        
        return {
            'macd_line': macd_line,
            'signal_line': signal_line,
            'histogram': histogram
        }

    def bollinger_bands(self, 
                       data: pd.Series, 
                       window: int = 20, 
                       num_std: float = 2.0) -> Dict[str, pd.Series]:
        """
        Calculate Bollinger Bands
        Middle Band = n-period SMA
        Upper Band = Middle Band + (k * n-period std)
        Lower Band = Middle Band - (k * n-period std)
        
        Parameters:
        -----------
        data : pd.Series
            Price series (typically close prices)
        window : int
            Lookback window in periods
        num_std : float
            Number of standard deviations for bands
            
        Returns:
        --------
        Dict[str, pd.Series]
            Dictionary with upper, middle, and lower bands
        """
        # Calculate middle band (SMA)
        middle_band = self.simple_moving_average(data, window=window)
        
        # Calculate standard deviation
        std = data.rolling(window=window).std()
        
        # Calculate upper and lower bands
        upper_band = middle_band + (num_std * std)
        lower_band = middle_band - (num_std * std)
        
        return {
            'upper': upper_band,
            'middle': middle_band,
            'lower': lower_band
        }

    def average_true_range(self, 
                          high: pd.Series, 
                          low: pd.Series, 
                          close: pd.Series, 
                          window: int = 14) -> pd.Series:
        """
        Calculate Average True Range (ATR)
        TR = max(high - low, abs(high - prev_close), abs(low - prev_close))
        ATR = n-period SMA of TR
        
        Parameters:
        -----------
        high : pd.Series
            High prices
        low : pd.Series
            Low prices
        close : pd.Series
            Close prices
        window : int
            Lookback window in periods
            
        Returns:
        --------
        pd.Series
            ATR values
        """
        # Calculate previous close
        prev_close = close.shift(1)
        
        # Calculate true range
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        
        true_range = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
        
        # Calculate ATR
        atr = true_range.rolling(window=window).mean()
        
        return atr

    def donchian_channel(self, 
                        high: pd.Series, 
                        low: pd.Series, 
                        window: int = 20) -> Dict[str, pd.Series]:
        """
        Calculate Donchian Channel
        Upper Band = n-period high
        Lower Band = n-period low
        Middle Band = (Upper Band + Lower Band) / 2
        
        Parameters:
        -----------
        high : pd.Series
            High prices
        low : pd.Series
            Low prices
        window : int
            Lookback window in periods
            
        Returns:
        --------
        Dict[str, pd.Series]
            Dictionary with upper, middle, and lower bands
        """
        # Calculate upper and lower bands
        upper_band = high.rolling(window=window).max()
        lower_band = low.rolling(window=window).min()
        
        # Calculate middle band
        middle_band = (upper_band + lower_band) / 2
        
        return {
            'upper': upper_band,
            'middle': middle_band,
            'lower': lower_band
        }

    def on_balance_volume(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """
        Calculate On Balance Volume (OBV)
        If close > prev_close: OBV = prev_OBV + volume
        If close < prev_close: OBV = prev_OBV - volume
        If close = prev_close: OBV = prev_OBV
        
        Parameters:
        -----------
        close : pd.Series
            Close prices
        volume : pd.Series
            Volume values
            
        Returns:
        --------
        pd.Series
            OBV values
        """
        # Calculate price change direction
        direction = np.where(close > close.shift(1), 1, np.where(close < close.shift(1), -1, 0))
        
        # Calculate OBV
        obv = (direction * volume).cumsum()
        
        return pd.Series(obv, index=close.index)

    def volume_price_trend(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """
        Calculate Volume Price Trend (VPT)
        VPT = prev_VPT + volume * (current_close - prev_close) / prev_close
        
        Parameters:
        -----------
        close : pd.Series
            Close prices
        volume : pd.Series
            Volume values
            
        Returns:
        --------
        pd.Series
            VPT values
        """
        # Calculate percentage price change
        price_change = close.pct_change()
        
        # Calculate VPT increments
        vpt_increments = volume * price_change
        
        # Calculate cumulative VPT
        vpt = vpt_increments.cumsum()
        
        return vpt

    def money_flow_index(self, 
                        high: pd.Series, 
                        low: pd.Series, 
                        close: pd.Series, 
                        volume: pd.Series, 
                        window: int = 14) -> pd.Series:
        """
        Calculate Money Flow Index (MFI)
        MFI = 100 - (100 / (1 + Money Flow Ratio))
        
        Parameters:
        -----------
        high : pd.Series
            High prices
        low : pd.Series
            Low prices
        close : pd.Series
            Close prices
        volume : pd.Series
            Volume values
        window : int
            Lookback window in periods
            
        Returns:
        --------
        pd.Series
            MFI values (0-100)
        """
        # Calculate typical price
        typical_price = (high + low + close) / 3
        
        # Calculate raw money flow
        raw_money_flow = typical_price * volume
        
        # Calculate money flow direction
        direction = np.where(typical_price > typical_price.shift(1), 1, -1)
        
        # Calculate positive and negative money flow
        positive_money_flow = np.where(direction > 0, raw_money_flow, 0)
        negative_money_flow = np.where(direction < 0, raw_money_flow, 0)
        
        # Calculate sum of positive and negative money flow over window
        positive_sum = pd.Series(positive_money_flow).rolling(window=window).sum()
        negative_sum = pd.Series(negative_money_flow).rolling(window=window).sum()
        
        # Calculate money flow ratio
        money_flow_ratio = positive_sum / negative_sum
        
        # Calculate MFI
        mfi = 100 - (100 / (1 + money_flow_ratio))
        
        return pd.Series(mfi, index=close.index)

    def calculate_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all technical indicators for the given dataframe
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with OHLCV data
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with all indicators
        """
        # Make sure we have the necessary columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Create a result dataframe with the same index as input
        result = pd.DataFrame(index=df.index)
        
        # Add original OHLCV data
        for col in required_columns:
            result[col] = df[col]
        
        # Moving Averages
        result['sma_20'] = self.simple_moving_average(df['close'], window=20)
        result['ema_20'] = self.exponential_moving_average(df['close'], span=20)
        result['dema_20'] = self.double_exponential_moving_average(df['close'], span=20)
        result['tema_20'] = self.triple_exponential_moving_average(df['close'], span=20)
        
        # Momentum Indicators
        result['rsi_14'] = self.relative_strength_index(df['close'], window=14)
        
        stoch = self.stochastic_oscillator(df['high'], df['low'], df['close'])
        result['stoch_k'] = stoch['K']
        result['stoch_d'] = stoch['D']
        
        result['roc_10'] = self.rate_of_change(df['close'], window=10)
        
        # Trend-Momentum Indicators
        macd_result = self.macd(df['close'])
        result['macd_line'] = macd_result['macd_line']
        result['macd_signal'] = macd_result['signal_line']
        result['macd_hist'] = macd_result['histogram']
        
        # Volatility Indicators
        bb = self.bollinger_bands(df['close'])
        result['bb_upper'] = bb['upper']
        result['bb_middle'] = bb['middle']
        result['bb_lower'] = bb['lower']
        
        result['atr_14'] = self.average_true_range(df['high'], df['low'], df['close'])
        
        dc = self.donchian_channel(df['high'], df['low'])
        result['dc_upper'] = dc['upper']
        result['dc_middle'] = dc['middle']
        result['dc_lower'] = dc['lower']
        
        # Volume Indicators
        result['obv'] = self.on_balance_volume(df['close'], df['volume'])
        result['vpt'] = self.volume_price_trend(df['close'], df['volume'])
        result['mfi_14'] = self.money_flow_index(df['high'], df['low'], df['close'], df['volume'])
        
        return result
