import os
import sys
import argparse
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import json
import random
from pathlib import Path

# Add parent directory to path to import from examples
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

try:
    from examples.chart_analysis import analyze_candlestick_chart, SurveyModel
except ImportError:
    print("Error importing chart_analysis module. Make sure you're running from the correct directory.")
    sys.exit(1)

def generate_mock_chart(output_path=None):
    """Generate a mock candlestick chart for testing"""
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Generate random price data
    np.random.seed(42)  # For reproducibility
    
    # Starting price
    price = 10000
    
    # Generate OHLC data
    dates = np.arange(50)
    opens = []
    highs = []
    lows = []
    closes = []
    
    for i in range(50):
        # Slight drift upward
        drift = np.random.normal(0.001, 0.01)
        
        # Random open
        open_price = price * (1 + np.random.normal(0, 0.005))
        opens.append(open_price)
        
        # Random close (with drift)
        close_price = open_price * (1 + drift + np.random.normal(0, 0.01))
        closes.append(close_price)
        
        # Random high and low
        high_price = max(open_price, close_price) * (1 + abs(np.random.normal(0, 0.005)))
        low_price = min(open_price, close_price) * (1 - abs(np.random.normal(0, 0.005)))
        
        highs.append(high_price)
        lows.append(low_price)
        
        # Update price for next iteration
        price = close_price
    
    # Create candlestick chart
    width = 0.8
    up_color = 'green'
    down_color = 'red'
    
    for i in range(len(dates)):
        # Determine color based on price movement
        color = up_color if closes[i] >= opens[i] else down_color
        
        # Plot vertical line for high-low
        ax.plot([dates[i], dates[i]], [lows[i], highs[i]], color=color, linewidth=1)
        
        # Plot candle body
        ax.bar(dates[i], abs(closes[i] - opens[i]), width, 
               bottom=min(opens[i], closes[i]), color=color, alpha=0.7)
    
    # Set labels and title
    ax.set_title('BTC/USDT Candlestick Chart')
    ax.set_xlabel('Time')
    ax.set_ylabel('Price (USD)')
    
    # Set y-axis to show prices correctly
    y_min = min(lows) * 0.995
    y_max = max(highs) * 1.005
    ax.set_ylim(y_min, y_max)
    
    # Remove x-ticks for cleaner look
    ax.set_xticks([])
    
    # Add some grid lines
    ax.grid(alpha=0.2)
    
    # Tight layout
    plt.tight_layout()
    
    # Save or show
    if output_path:
        plt.savefig(output_path)
        print(f"Mock chart saved to: {output_path}")
    else:
        plt.show()
    
    plt.close(fig)

def demo_simplified():
    """Run a simplified demo of the candlestick chart analysis"""
    # First, generate the survey JSON if it doesn't exist
    survey_json_path = os.path.join(current_dir, "cryptocurrency_prediction_survey_machine_readable.json")
    
    if not os.path.exists(survey_json_path):
        try:
            # Try to generate the survey JSON file
            from examples.generate_survey_json import generate_survey_json
            
            survey_data = generate_survey_json()
            with open(survey_json_path, 'w') as f:
                json.dump(survey_data, f, indent=2)
            print(f"Generated survey JSON file: {survey_json_path}")
        except ImportError:
            print("Warning: Could not generate survey JSON file.")
    
    # Create a test directory if it doesn't exist
    test_dir = os.path.join(current_dir, "test_data")
    os.makedirs(test_dir, exist_ok=True)
    
    # Generate a mock chart
    chart_path = os.path.join(test_dir, "mock_btc_chart.png")
    generate_mock_chart(chart_path)
    
    # Load the chart image
    img = Image.open(chart_path)
    img_array = np.array(img)
    
    print("\n========== CANDLESTICK CHART ANALYSIS ==========")
    print(f"Analyzing chart: {chart_path}")
    
    # Timeframes to test
    timeframes = ['15m', '1h', '4h', '1d', '7d']
    
    for timeframe in timeframes:
        print(f"\n----- Timeframe: {timeframe} -----")
        
        # Analyze with chart analysis
        chart_result = analyze_candlestick_chart(img_array, timeframe)
        
        # Create a survey model
        try:
            survey_model = SurveyModel()
        except Exception as e:
            from examples.chart_analysis import ChartAnalyzer
            class MockSurveyModel:
                def process(self, timeframe, user_input="", detected_patterns=None):
                    return {
                        "prediction": random.choice(["BUY", "SELL", "HOLD", "UPTREND", "DOWNTREND"]),
                        "confidence": round(0.6 + random.random() * 0.3, 2)
                    }
            survey_model = MockSurveyModel()
        
        # Get survey prediction
        survey_result = survey_model.process(
            timeframe,
            "I think the market looks bullish",
            chart_result["detected_patterns"]
        )
        
        # Print results
        print("\nChart Analysis:")
        print(f"- Prediction: {chart_result['prediction']}")
        print(f"- Confidence: {chart_result['confidence']:.2f}")
        print(f"- Detected patterns: {', '.join(chart_result['detected_patterns'])}")
        print("\nKey Technical Indicators:")
        for name, value in chart_result['technical_indicators'].items():
            if name in ['RSI', 'MACD', 'Bollinger Bands', 'Stochastic']:
                print(f"- {name}: {value}")
        
        print("\nSurvey Model:")
        print(f"- Prediction: {survey_result['prediction']}")
        print(f"- Confidence: {survey_result['confidence']:.2f}")
        
        # Combined prediction
        combined_confidence = (chart_result["confidence"] + survey_result["confidence"]) / 2
        
        # Determine combined prediction based on weighted confidence
        if chart_result["confidence"] > survey_result["confidence"]:
            combined_prediction = chart_result["prediction"]
        else:
            combined_prediction = survey_result["prediction"]
        
        # If predictions disagree but confidences are close, use NEUTRAL
        if (chart_result["prediction"] != survey_result["prediction"] and 
            abs(chart_result["confidence"] - survey_result["confidence"]) < 0.1):
            combined_prediction = "NEUTRAL"
        
        print("\nCombined Analysis:")
        print(f"- Prediction: {combined_prediction}")
        print(f"- Confidence: {combined_confidence:.2f}")

if __name__ == "__main__":
    demo_simplified()
