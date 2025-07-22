import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple


class CandlestickPatterns:
    """
    Implementation of candlestick pattern recognition algorithms
    """

    def __init__(self):
        # Threshold values for pattern detection
        self.doji_threshold = 0.05  # Max ratio of body to range for Doji
        self.hammer_threshold = 0.3  # Max ratio of body to range for Hammer
        self.long_body_threshold = 0.7  # Min ratio of body to range for long body
        self.engulfing_threshold = 1.2  # Min ratio for engulfing patterns
        
    def _get_candle_features(self, ohlc: pd.DataFrame, idx: int) -> Dict[str, float]:
        """
        Extract features from a single candlestick
        
        Parameters:
        -----------
        ohlc : pd.DataFrame
            DataFrame with 'open', 'high', 'low', 'close' columns
        idx : int
            Index of the candlestick to analyze
            
        Returns:
        --------
        Dict[str, float]
            Dictionary of candlestick features
        """
        # Get OHLC values
        o = ohlc['open'].iloc[idx]
        h = ohlc['high'].iloc[idx]
        l = ohlc['low'].iloc[idx]
        c = ohlc['close'].iloc[idx]
        
        # Calculate candlestick features
        is_bullish = c > o
        body = abs(c - o)
        range_size = h - l
        body_percent = body / range_size if range_size > 0 else 0
        
        upper_wick = h - max(o, c)
        lower_wick = min(o, c) - l
        upper_wick_percent = upper_wick / range_size if range_size > 0 else 0
        lower_wick_percent = lower_wick / range_size if range_size > 0 else 0
        
        return {
            'open': o,
            'high': h,
            'low': l,
            'close': c,
            'is_bullish': is_bullish,
            'body': body,
            'range_size': range_size,
            'body_percent': body_percent,
            'upper_wick': upper_wick,
            'lower_wick': lower_wick,
            'upper_wick_percent': upper_wick_percent,
            'lower_wick_percent': lower_wick_percent
        }
    
    def detect_doji(self, ohlc: pd.DataFrame, idx: int) -> bool:
        """
        Detect Doji pattern (small body with wicks on both sides)
        
        Parameters:
        -----------
        ohlc : pd.DataFrame
            DataFrame with 'open', 'high', 'low', 'close' columns
        idx : int
            Index of the candlestick to analyze
            
        Returns:
        --------
        bool
            True if Doji pattern is detected
        """
        features = self._get_candle_features(ohlc, idx)
        
        # Doji has very small body compared to the range
        return features['body_percent'] < self.doji_threshold

    def detect_hammer(self, ohlc: pd.DataFrame, idx: int) -> Tuple[bool, str]:
        """
        Detect Hammer or Shooting Star patterns
        
        Parameters:
        -----------
        ohlc : pd.DataFrame
            DataFrame with 'open', 'high', 'low', 'close' columns
        idx : int
            Index of the candlestick to analyze
            
        Returns:
        --------
        Tuple[bool, str]
            (is_detected, pattern_name)
        """
        features = self._get_candle_features(ohlc, idx)
        
        # Small body
        small_body = features['body_percent'] < self.hammer_threshold
        
        # Check for Hammer (small body, little/no upper wick, long lower wick)
        if (small_body and 
            features['upper_wick_percent'] < 0.1 and 
            features['lower_wick_percent'] > 0.6):
            
            # Bullish Hammer in downtrend
            if self._is_downtrend(ohlc, idx):
                return True, "Hammer"
            # Hanging Man in uptrend
            elif self._is_uptrend(ohlc, idx):
                return True, "Hanging Man"
        
        # Check for Shooting Star (small body, little/no lower wick, long upper wick)
        if (small_body and
            features['lower_wick_percent'] < 0.1 and
            features['upper_wick_percent'] > 0.6):
            
            # Shooting Star in uptrend
            if self._is_uptrend(ohlc, idx):
                return True, "Shooting Star"
            # Inverted Hammer in downtrend
            elif self._is_downtrend(ohlc, idx):
                return True, "Inverted Hammer"
                
        return False, ""

    def detect_spinning_top(self, ohlc: pd.DataFrame, idx: int) -> bool:
        """
        Detect Spinning Top pattern (small body with significant wicks on both sides)
        
        Parameters:
        -----------
        ohlc : pd.DataFrame
            DataFrame with 'open', 'high', 'low', 'close' columns
        idx : int
            Index of the candlestick to analyze
            
        Returns:
        --------
        bool
            True if Spinning Top pattern is detected
        """
        features = self._get_candle_features(ohlc, idx)
        
        # Spinning Top has small body and significant wicks on both sides
        return (features['body_percent'] < 0.3 and
                features['upper_wick_percent'] > 0.2 and
                features['lower_wick_percent'] > 0.2)

    def detect_marubozu(self, ohlc: pd.DataFrame, idx: int) -> Tuple[bool, str]:
        """
        Detect Marubozu pattern (long body with little/no wicks)
        
        Parameters:
        -----------
        ohlc : pd.DataFrame
            DataFrame with 'open', 'high', 'low', 'close' columns
        idx : int
            Index of the candlestick to analyze
            
        Returns:
        --------
        Tuple[bool, str]
            (is_detected, pattern_name)
        """
        features = self._get_candle_features(ohlc, idx)
        
        # Marubozu has long body and little/no wicks
        is_marubozu = (features['body_percent'] > self.long_body_threshold and
                       features['upper_wick_percent'] < 0.05 and
                       features['lower_wick_percent'] < 0.05)
        
        if is_marubozu:
            pattern_name = "Bullish Marubozu" if features['is_bullish'] else "Bearish Marubozu"
            return True, pattern_name
        
        return False, ""

    def detect_engulfing(self, ohlc: pd.DataFrame, idx: int) -> Tuple[bool, str]:
        """
        Detect Engulfing pattern (current candle's body completely engulfs previous candle's body)
        
        Parameters:
        -----------
        ohlc : pd.DataFrame
            DataFrame with 'open', 'high', 'low', 'close' columns
        idx : int
            Index of the candlestick to analyze
            
        Returns:
        --------
        Tuple[bool, str]
            (is_detected, pattern_name)
        """
        # Need at least 2 candles
        if idx < 1:
            return False, ""
        
        # Get features for current and previous candles
        curr = self._get_candle_features(ohlc, idx)
        prev = self._get_candle_features(ohlc, idx - 1)
        
        # Check for Bullish Engulfing (current is bullish, previous is bearish)
        if (curr['is_bullish'] and not prev['is_bullish'] and
            curr['body'] > prev['body'] * self.engulfing_threshold and
            curr['open'] <= prev['close'] and
            curr['close'] >= prev['open']):
            return True, "Bullish Engulfing"
        
        # Check for Bearish Engulfing (current is bearish, previous is bullish)
        if (not curr['is_bullish'] and prev['is_bullish'] and
            curr['body'] > prev['body'] * self.engulfing_threshold and
            curr['open'] >= prev['close'] and
            curr['close'] <= prev['open']):
            return True, "Bearish Engulfing"
        
        return False, ""

    def detect_harami(self, ohlc: pd.DataFrame, idx: int) -> Tuple[bool, str]:
        """
        Detect Harami pattern (current candle's body completely inside previous candle's body)
        
        Parameters:
        -----------
        ohlc : pd.DataFrame
            DataFrame with 'open', 'high', 'low', 'close' columns
        idx : int
            Index of the candlestick to analyze
            
        Returns:
        --------
        Tuple[bool, str]
            (is_detected, pattern_name)
        """
        # Need at least 2 candles
        if idx < 1:
            return False, ""
        
        # Get features for current and previous candles
        curr = self._get_candle_features(ohlc, idx)
        prev = self._get_candle_features(ohlc, idx - 1)
        
        # Check for Bullish Harami (previous is bearish, current is bullish)
        if (curr['is_bullish'] and not prev['is_bullish'] and
            curr['body'] < prev['body'] * 0.6 and
            min(curr['open'], curr['close']) > min(prev['open'], prev['close']) and
            max(curr['open'], curr['close']) < max(prev['open'], prev['close'])):
            return True, "Bullish Harami"
        
        # Check for Bearish Harami (previous is bullish, current is bearish)
        if (not curr['is_bullish'] and prev['is_bullish'] and
            curr['body'] < prev['body'] * 0.6 and
            min(curr['open'], curr['close']) > min(prev['open'], prev['close']) and
            max(curr['open'], curr['close']) < max(prev['open'], prev['close'])):
            return True, "Bearish Harami"
        
        return False, ""

    def detect_morning_star(self, ohlc: pd.DataFrame, idx: int) -> bool:
        """
        Detect Morning Star pattern (three-candle bullish reversal pattern)
        
        Parameters:
        -----------
        ohlc : pd.DataFrame
            DataFrame with 'open', 'high', 'low', 'close' columns
        idx : int
            Index of the candlestick to analyze
            
        Returns:
        --------
        bool
            True if Morning Star pattern is detected
        """
        # Need at least 3 candles
        if idx < 2:
            return False
        
        # Get features for the three candles
        first = self._get_candle_features(ohlc, idx - 2)  # First candle (bearish)
        middle = self._get_candle_features(ohlc, idx - 1)  # Middle candle (small body)
        last = self._get_candle_features(ohlc, idx)  # Last candle (bullish)
        
        # Check pattern criteria:
        # 1. First candle is bearish with long body
        # 2. Middle candle has small body (could be either bullish or bearish)
        # 3. Last candle is bullish with body closing at least halfway up the first candle's body
        # 4. Gap down between first and middle
        # 5. Gap up between middle and last
        
        is_pattern = (
            # First candle is bearish with long body
            not first['is_bullish'] and first['body_percent'] > 0.6 and
            
            # Middle candle has small body
            middle['body_percent'] < 0.3 and
            
            # Last candle is bullish
            last['is_bullish'] and
            
            # Gap down between first and middle
            max(middle['open'], middle['close']) < first['close'] and
            
            # Gap up between middle and last
            min(last['open'], last['close']) > max(middle['open'], middle['close']) and
            
            # Last candle closes at least halfway up the first candle's body
            last['close'] > (first['open'] + first['close']) / 2
        )
        
        return is_pattern

    def detect_evening_star(self, ohlc: pd.DataFrame, idx: int) -> bool:
        """
        Detect Evening Star pattern (three-candle bearish reversal pattern)
        
        Parameters:
        -----------
        ohlc : pd.DataFrame
            DataFrame with 'open', 'high', 'low', 'close' columns
        idx : int
            Index of the candlestick to analyze
            
        Returns:
        --------
        bool
            True if Evening Star pattern is detected
        """
        # Need at least 3 candles
        if idx < 2:
            return False
        
        # Get features for the three candles
        first = self._get_candle_features(ohlc, idx - 2)  # First candle (bullish)
        middle = self._get_candle_features(ohlc, idx - 1)  # Middle candle (small body)
        last = self._get_candle_features(ohlc, idx)  # Last candle (bearish)
        
        # Check pattern criteria:
        # 1. First candle is bullish with long body
        # 2. Middle candle has small body (could be either bullish or bearish)
        # 3. Last candle is bearish with body closing at least halfway down the first candle's body
        # 4. Gap up between first and middle
        # 5. Gap down between middle and last
        
        is_pattern = (
            # First candle is bullish with long body
            first['is_bullish'] and first['body_percent'] > 0.6 and
            
            # Middle candle has small body
            middle['body_percent'] < 0.3 and
            
            # Last candle is bearish
            not last['is_bullish'] and
            
            # Gap up between first and middle
            min(middle['open'], middle['close']) > first['close'] and
            
            # Gap down between middle and last
            max(last['open'], last['close']) < min(middle['open'], middle['close']) and
            
            # Last candle closes at least halfway down the first candle's body
            last['close'] < (first['open'] + first['close']) / 2
        )
        
        return is_pattern

    def detect_tweezer_top(self, ohlc: pd.DataFrame, idx: int) -> bool:
        """
        Detect Tweezer Top pattern (two-candle bearish reversal pattern)
        
        Parameters:
        -----------
        ohlc : pd.DataFrame
            DataFrame with 'open', 'high', 'low', 'close' columns
        idx : int
            Index of the candlestick to analyze
            
        Returns:
        --------
        bool
            True if Tweezer Top pattern is detected
        """
        # Need at least 2 candles
        if idx < 1:
            return False
        
        # Get features for current and previous candles
        curr = self._get_candle_features(ohlc, idx)
        prev = self._get_candle_features(ohlc, idx - 1)
        
        # Check pattern criteria:
        # 1. First candle is bullish
        # 2. Second candle is bearish
        # 3. Both candles have similar highs (tops)
        # 4. In an uptrend
        
        is_pattern = (
            prev['is_bullish'] and
            not curr['is_bullish'] and
            abs(curr['high'] - prev['high']) / ((curr['high'] + prev['high']) / 2) < 0.002 and  # Similar highs
            self._is_uptrend(ohlc, idx - 1)
        )
        
        return is_pattern

    def detect_tweezer_bottom(self, ohlc: pd.DataFrame, idx: int) -> bool:
        """
        Detect Tweezer Bottom pattern (two-candle bullish reversal pattern)
        
        Parameters:
        -----------
        ohlc : pd.DataFrame
            DataFrame with 'open', 'high', 'low', 'close' columns
        idx : int
            Index of the candlestick to analyze
            
        Returns:
        --------
        bool
            True if Tweezer Bottom pattern is detected
        """
        # Need at least 2 candles
        if idx < 1:
            return False
        
        # Get features for current and previous candles
        curr = self._get_candle_features(ohlc, idx)
        prev = self._get_candle_features(ohlc, idx - 1)
        
        # Check pattern criteria:
        # 1. First candle is bearish
        # 2. Second candle is bullish
        # 3. Both candles have similar lows (bottoms)
        # 4. In a downtrend
        
        is_pattern = (
            not prev['is_bullish'] and
            curr['is_bullish'] and
            abs(curr['low'] - prev['low']) / ((curr['low'] + prev['low']) / 2) < 0.002 and  # Similar lows
            self._is_downtrend(ohlc, idx - 1)
        )
        
        return is_pattern

    def _is_uptrend(self, ohlc: pd.DataFrame, idx: int, lookback: int = 5) -> bool:
        """
        Determine if the market is in an uptrend
        
        Parameters:
        -----------
        ohlc : pd.DataFrame
            DataFrame with 'open', 'high', 'low', 'close' columns
        idx : int
            Current index
        lookback : int
            Number of periods to look back
            
        Returns:
        --------
        bool
            True if in uptrend
        """
        # Check if we have enough data
        start_idx = max(0, idx - lookback)
        
        if start_idx == idx:
            return False
        
        # Calculate simple moving average
        prices = ohlc['close'].iloc[start_idx:idx+1]
        
        # If the current price is above the average of the lookback period
        return ohlc['close'].iloc[idx] > prices.mean()

    def _is_downtrend(self, ohlc: pd.DataFrame, idx: int, lookback: int = 5) -> bool:
        """
        Determine if the market is in a downtrend
        
        Parameters:
        -----------
        ohlc : pd.DataFrame
            DataFrame with 'open', 'high', 'low', 'close' columns
        idx : int
            Current index
        lookback : int
            Number of periods to look back
            
        Returns:
        --------
        bool
            True if in downtrend
        """
        # Check if we have enough data
        start_idx = max(0, idx - lookback)
        
        if start_idx == idx:
            return False
        
        # Calculate simple moving average
        prices = ohlc['close'].iloc[start_idx:idx+1]
        
        # If the current price is below the average of the lookback period
        return ohlc['close'].iloc[idx] < prices.mean()

    def analyze_all_patterns(self, ohlc: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze all candlestick patterns for each row in the dataframe
        
        Parameters:
        -----------
        ohlc : pd.DataFrame
            DataFrame with 'open', 'high', 'low', 'close' columns
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with pattern detection results
        """
        # Initialize results dataframe
        results = pd.DataFrame(index=ohlc.index)
        
        # Single-candle patterns
        results['doji'] = False
        results['hammer'] = False
        results['shooting_star'] = False
        results['inverted_hammer'] = False
        results['hanging_man'] = False
        results['spinning_top'] = False
        results['marubozu_bullish'] = False
        results['marubozu_bearish'] = False
        
        # Multi-candle patterns
        results['engulfing_bullish'] = False
        results['engulfing_bearish'] = False
        results['harami_bullish'] = False
        results['harami_bearish'] = False
        results['morning_star'] = False
        results['evening_star'] = False
        results['tweezer_top'] = False
        results['tweezer_bottom'] = False
        
        # Analyze each candle
        for i in range(len(ohlc)):
            # Single-candle patterns
            results.iloc[i, results.columns.get_loc('doji')] = self.detect_doji(ohlc, i)
            
            hammer_detected, hammer_type = self.detect_hammer(ohlc, i)
            if hammer_detected:
                if hammer_type == "Hammer":
                    results.iloc[i, results.columns.get_loc('hammer')] = True
                elif hammer_type == "Shooting Star":
                    results.iloc[i, results.columns.get_loc('shooting_star')] = True
                elif hammer_type == "Inverted Hammer":
                    results.iloc[i, results.columns.get_loc('inverted_hammer')] = True
                elif hammer_type == "Hanging Man":
                    results.iloc[i, results.columns.get_loc('hanging_man')] = True
            
            results.iloc[i, results.columns.get_loc('spinning_top')] = self.detect_spinning_top(ohlc, i)
            
            marubozu_detected, marubozu_type = self.detect_marubozu(ohlc, i)
            if marubozu_detected:
                if marubozu_type == "Bullish Marubozu":
                    results.iloc[i, results.columns.get_loc('marubozu_bullish')] = True
                elif marubozu_type == "Bearish Marubozu":
                    results.iloc[i, results.columns.get_loc('marubozu_bearish')] = True
            
            # Multi-candle patterns (need sufficient history)
            if i >= 1:
                engulfing_detected, engulfing_type = self.detect_engulfing(ohlc, i)
                if engulfing_detected:
                    if engulfing_type == "Bullish Engulfing":
                        results.iloc[i, results.columns.get_loc('engulfing_bullish')] = True
                    elif engulfing_type == "Bearish Engulfing":
                        results.iloc[i, results.columns.get_loc('engulfing_bearish')] = True
                
                harami_detected, harami_type = self.detect_harami(ohlc, i)
                if harami_detected:
                    if harami_type == "Bullish Harami":
                        results.iloc[i, results.columns.get_loc('harami_bullish')] = True
                    elif harami_type == "Bearish Harami":
                        results.iloc[i, results.columns.get_loc('harami_bearish')] = True
                
                results.iloc[i, results.columns.get_loc('tweezer_top')] = self.detect_tweezer_top(ohlc, i)
                results.iloc[i, results.columns.get_loc('tweezer_bottom')] = self.detect_tweezer_bottom(ohlc, i)
            
            # Three-candle patterns (need more history)
            if i >= 2:
                results.iloc[i, results.columns.get_loc('morning_star')] = self.detect_morning_star(ohlc, i)
                results.iloc[i, results.columns.get_loc('evening_star')] = self.detect_evening_star(ohlc, i)
        
        return results
        
    def detect_patterns_from_image(self, image_features: Dict[str, float]) -> List[str]:
        """
        Detect potential candlestick patterns based on image features
        
        Parameters:
        -----------
        image_features : Dict[str, float]
            Dictionary of image features
            
        Returns:
        --------
        List[str]
            List of potential candlestick patterns
        """
        patterns = []
        
        # Simplified pattern detection logic based on image features
        mean = image_features['mean']
        std = image_features['std']
        contrast = image_features['contrast']
        grad_mean = image_features['grad_mean']
        
        # Use threshold values that won't change between runs
        # These strict threshold values help ensure deterministic results
        if mean > 0.5 and std < 0.2:
            patterns.append("Bullish Pattern")
            if contrast > 0.7:
                patterns.append("Potential Engulfing")
            if grad_mean > 0.05:
                patterns.append("Potential Hammer")
        elif mean < 0.4:
            patterns.append("Bearish Pattern")
            if contrast > 0.7:
                patterns.append("Potential Evening Star")
            if grad_mean > 0.05:
                patterns.append("Potential Shooting Star")
        else:
            patterns.append("Neutral Pattern")
            if std < 0.1:
                patterns.append("Potential Doji")
            elif contrast < 0.3:
                patterns.append("Consolidation")
        
        # Ensure deterministic order of patterns
        patterns.sort()
        
        return patterns
