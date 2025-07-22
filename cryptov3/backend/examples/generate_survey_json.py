import json
import sys
import os
import argparse

def generate_survey_json():
    """
    Generate a machine-readable JSON file based on the survey content
    """
    survey_data = {
        "title": "Cryptocurrency Price Prediction Algorithms: A Survey and Future Directions",
        "authors": [
            "David L. John",
            "Sebastian Binnewies",
            "Bela Stantic"
        ],
        "parameters": {
            "technical_indicators": [
                "RSI",
                "MACD",
                "Moving Averages",
                "Bollinger Bands",
                "Fibonacci Retracements",
                "Stochastic Oscillator",
                "Volume"
            ],
            "blockchain_features": [
                "Transaction Fees",
                "Hash Rate",
                "Transaction Volume",
                "Miners Revenue",
                "Transaction Rate",
                "Mining Difficulty",
                "Block Size"
            ],
            "sentiment_analysis": [
                "Twitter Sentiment",
                "Reddit Sentiment",
                "News Articles",
                "Social Media Volume"
            ]
        },
        "algorithms": {
            "LSTM": {
                "accuracy": 0.78,
                "effectiveness": "high",
                "popular_parameters": ["price", "volume"]
            },
            "GRU": {
                "accuracy": 0.76,
                "effectiveness": "high",
                "popular_parameters": ["price", "volume"]
            },
            "CNN": {
                "accuracy": 0.75,
                "effectiveness": "medium-high",
                "popular_parameters": ["technical indicators", "price charts"]
            },
            "LSTM-GRU": {
                "accuracy": 0.85,
                "effectiveness": "very high",
                "popular_parameters": ["price", "volume", "blockchain features"]
            },
            "CNN-LSTM": {
                "accuracy": 0.82,
                "effectiveness": "high",
                "popular_parameters": ["price charts", "technical indicators"]
            },
            "ConvLSTM": {
                "accuracy": 0.83,
                "effectiveness": "high",
                "popular_parameters": ["price charts", "technical indicators"]
            },
            "MRC-LSTM": {
                "accuracy": 0.84,
                "effectiveness": "high",
                "popular_parameters": ["price", "volume", "external factors"]
            },
            "Transformers": {
                "accuracy": 0.88,
                "effectiveness": "very high",
                "popular_parameters": ["price", "volume", "sentiment", "technical indicators"]
            }
        },
        "timeframes": {
            "15m": {
                "predictability": "low",
                "volatility": "very high",
                "recommended_algorithms": ["CNN-LSTM", "LSTM-GRU"]
            },
            "1h": {
                "predictability": "medium-low",
                "volatility": "high",
                "recommended_algorithms": ["LSTM-GRU", "CNN-LSTM"]
            },
            "4h": {
                "predictability": "medium",
                "volatility": "medium",
                "recommended_algorithms": ["LSTM-GRU", "MRC-LSTM"]
            },
            "1d": {
                "predictability": "medium-high",
                "volatility": "medium-low",
                "recommended_algorithms": ["Transformers", "LSTM-GRU", "MRC-LSTM"]
            },
            "7d": {
                "predictability": "high",
                "volatility": "low",
                "recommended_algorithms": ["Transformers", "LSTM-GRU"]
            }
        },
        "patterns": {
            "bullish": [
                "Bullish Engulfing",
                "Hammer",
                "Morning Star",
                "Three White Soldiers",
                "Piercing Line"
            ],
            "bearish": [
                "Bearish Engulfing",
                "Hanging Man",
                "Evening Star",
                "Three Black Crows",
                "Dark Cloud Cover"
            ],
            "neutral": [
                "Doji",
                "Spinning Top",
                "Harami"
            ]
        },
        "research_gaps": [
            "Integration of technical indicators with blockchain features",
            "Real-world profitability analysis of prediction models",
            "Application of Transformer models to cryptocurrency markets",
            "Optimization of prediction algorithms for specific timeframes"
        ]
    }
    
    return survey_data

def main():
    parser = argparse.ArgumentParser(description="Generate machine-readable JSON from cryptocurrency survey")
    parser.add_argument('--output', default='cryptocurrency_prediction_survey_machine_readable.json', 
                        help='Output JSON file path')
    args = parser.parse_args()
    
    survey_data = generate_survey_json()
    
    with open(args.output, 'w') as f:
        json.dump(survey_data, f, indent=2)
    
    print(f"Generated machine-readable JSON file: {args.output}")

if __name__ == "__main__":
    main()
