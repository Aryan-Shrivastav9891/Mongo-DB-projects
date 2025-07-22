import { TimeframeOption } from './theme';

interface SurveyData {
  title: string;
  content?: string;
  parameters?: {
    technical_indicators?: string[];
    blockchain_features?: string[];
    sentiment_analysis?: string[];
  };
  algorithms?: Record<string, {
    accuracy?: number;
    effectiveness?: string;
  }>;
}

interface PredictionResponse {
  prediction: string;
  confidence: number;
  reasoning?: string[];
}

export class SurveyModelProcessor {
  private surveyData: SurveyData;
  
  constructor(surveyData?: SurveyData) {
    this.surveyData = surveyData || {
      title: "Cryptocurrency Price Prediction Algorithms: A Survey",
      parameters: {
        technical_indicators: ["RSI", "MACD", "Moving Averages", "Bollinger Bands"],
        blockchain_features: ["Hash Rate", "Transaction Volume", "Miner Revenue"],
        sentiment_analysis: ["Social Media", "News Articles"]
      },
      algorithms: {
        "LSTM-GRU": {
          accuracy: 0.85,
          effectiveness: "high"
        },
        "CNN-LSTM": {
          accuracy: 0.82,
          effectiveness: "medium-high"
        },
        "Transformers": {
          accuracy: 0.88,
          effectiveness: "high"
        }
      }
    };
  }

  public processUserInput(input: string): { sentiment: number; keywords: string[] } {
    // A very simple sentiment analyzer
    const positiveWords = [
      'bull', 'bullish', 'up', 'rise', 'rising', 'growth', 'grow', 'increasing',
      'positive', 'good', 'great', 'excellent', 'strong', 'higher', 'rally',
      'support', 'confidence', 'optimistic', 'opportunity', 'buy', 'accumulate'
    ];
    
    const negativeWords = [
      'bear', 'bearish', 'down', 'fall', 'falling', 'drop', 'dropping', 'decreasing',
      'negative', 'bad', 'weak', 'lower', 'decline', 'resistance', 'pessimistic',
      'fear', 'sell', 'dump', 'risk', 'crash', 'correction'
    ];
    
    const inputLower = input.toLowerCase();
    let sentiment = 0;
    const foundKeywords: string[] = [];
    
    // Check for positive words
    positiveWords.forEach(word => {
      if (inputLower.includes(word)) {
        sentiment += 1;
        foundKeywords.push(word);
      }
    });
    
    // Check for negative words
    negativeWords.forEach(word => {
      if (inputLower.includes(word)) {
        sentiment -= 1;
        foundKeywords.push(word);
      }
    });
    
    return { sentiment, keywords: foundKeywords };
  }

  public processDetectedPatterns(patterns: string[]): { bullishPatterns: number; bearishPatterns: number; neutralPatterns: number } {
    let bullishPatterns = 0;
    let bearishPatterns = 0;
    let neutralPatterns = 0;
    
    patterns.forEach(pattern => {
      const patternLower = pattern.toLowerCase();
      
      if (/bullish|hammer|morning star|white soldiers|piercing/i.test(patternLower)) {
        bullishPatterns++;
      } else if (/bearish|hanging|evening star|black crows|dark cloud/i.test(patternLower)) {
        bearishPatterns++;
      } else {
        neutralPatterns++;
      }
    });
    
    return { bullishPatterns, bearishPatterns, neutralPatterns };
  }
  
  public processTechnicalIndicators(indicators: Record<string, number | string>): { bullishSignals: number; bearishSignals: number } {
    let bullishSignals = 0;
    let bearishSignals = 0;
    
    // Process RSI
    const rsi = indicators.RSI as number;
    if (rsi !== undefined) {
      if (rsi < 30) bullishSignals++;  // Oversold
      else if (rsi > 70) bearishSignals++;  // Overbought
    }
    
    // Process MACD
    const macd = indicators.MACD as number;
    if (macd !== undefined) {
      if (macd > 0) bullishSignals++;  // Bullish
      else if (macd < 0) bearishSignals++;  // Bearish
    }
    
    // Process Bollinger Bands
    const bb = indicators['Bollinger Bands'] as string;
    if (bb !== undefined) {
      if (bb === 'Above') bearishSignals++;  // Overbought
      else if (bb === 'Below') bullishSignals++;  // Oversold
    }
    
    // Process Stochastic
    const stoch = indicators.Stochastic as number;
    if (stoch !== undefined) {
      if (stoch < 20) bullishSignals++;  // Oversold
      else if (stoch > 80) bearishSignals++;  // Overbought
    }
    
    return { bullishSignals, bearishSignals };
  }

  public predict(
    timeframe: TimeframeOption, 
    detectedPatterns: string[] = [],
    technicalIndicators: Record<string, number | string> = {},
    userInput: string = ''
  ): PredictionResponse {
    // Process user input for sentiment
    const { sentiment, keywords } = this.processUserInput(userInput);
    
    // Process detected patterns
    const { bullishPatterns, bearishPatterns } = this.processDetectedPatterns(detectedPatterns);
    
    // Process technical indicators
    const { bullishSignals, bearishSignals } = this.processTechnicalIndicators(technicalIndicators);
    
    // Combine signals
    const totalBullish = bullishPatterns + bullishSignals + (sentiment > 0 ? 1 : 0);
    const totalBearish = bearishPatterns + bearishSignals + (sentiment < 0 ? 1 : 0);
    
    // Adjust weights based on timeframe
    let timeframeMultiplier = 1.0;
    switch (timeframe) {
      case '15m':
        timeframeMultiplier = 1.2; // More volatile, more confidence
        break;
      case '1h':
        timeframeMultiplier = 1.1;
        break;
      case '4h':
        timeframeMultiplier = 1.0; // Neutral
        break;
      case '1d':
        timeframeMultiplier = 0.9;
        break;
      case '7d':
        timeframeMultiplier = 0.8; // Less volatile, less confidence
        break;
    }
    
    // Calculate confidence level (0.5-0.95)
    let confidence = 0.5;
    const signalDifference = Math.abs(totalBullish - totalBearish);
    
    if (signalDifference > 0) {
      // More signals = more confidence, but cap at 0.95
      confidence = Math.min(0.95, 0.6 + (signalDifference * 0.1));
    }
    
    // Apply timeframe multiplier
    confidence *= timeframeMultiplier;
    confidence = Math.min(0.95, confidence); // Cap at 0.95
    
    // Determine prediction
    let prediction: string;
    const reasoning: string[] = [];
    
    if (totalBullish > totalBearish * 1.2) {
      prediction = timeframe === '1d' || timeframe === '7d' ? 'UPTREND' : 'BUY';
      reasoning.push(`${bullishPatterns} bullish patterns detected`);
      reasoning.push(`${bullishSignals} bullish technical signals`);
      if (sentiment > 0) reasoning.push(`Positive sentiment in user input: ${keywords.join(', ')}`);
    } else if (totalBearish > totalBullish * 1.2) {
      prediction = timeframe === '1d' || timeframe === '7d' ? 'DOWNTREND' : 'SELL';
      reasoning.push(`${bearishPatterns} bearish patterns detected`);
      reasoning.push(`${bearishSignals} bearish technical signals`);
      if (sentiment < 0) reasoning.push(`Negative sentiment in user input: ${keywords.join(', ')}`);
    } else {
      prediction = 'NEUTRAL';
      reasoning.push('Mixed signals detected');
      reasoning.push(`${bullishPatterns} bullish vs ${bearishPatterns} bearish patterns`);
      reasoning.push(`${bullishSignals} bullish vs ${bearishSignals} bearish technical signals`);
    }
    
    return {
      prediction,
      confidence: parseFloat(confidence.toFixed(2)),
      reasoning
    };
  }
}
