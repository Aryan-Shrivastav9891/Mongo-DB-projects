import axios from 'axios';
import { TimeframeOption } from './theme';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export interface TechnicalIndicators {
  [key: string]: number | string;
}

export interface PredictionResponse {
  prediction: 'BUY' | 'SELL' | 'HOLD';
  confidence: number;
  detected_patterns: string[];
  technical_indicators: TechnicalIndicators;
  timestamp?: string;
}

export const predictCandlestickChart = async (
  imageFile: File,
  timeframe: TimeframeOption
): Promise<PredictionResponse> => {
    console.log('Predicting candlestick chart with file:', imageFile.name, 'and timeframe:', timeframe);
  const formData = new FormData();
  formData.append('image', imageFile);
  formData.append('timeframe', timeframe);

  try {
    const response = await axios.post<any>(
      `${API_URL}/api/predict`,
      formData,
      {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      }
    );
    
    // Log the raw response for debugging
    console.log('API raw response:', response.data);
    
    console.log('Transforming API response:', response.data);
    
    // Transform the response to match our interface if needed
    const transformedResponse: PredictionResponse = {
      prediction: response.data.prediction || 'HOLD',
      confidence: response.data.confidence || 0.5,
      detected_patterns: response.data.patterns || response.data.detected_patterns || [],
      technical_indicators: response.data.technical_indicators || {},
      timestamp: response.data.timestamp || new Date().toISOString()
    };
    
    return transformedResponse;
  } catch (error) {
    console.error('Error predicting candlestick chart:', error);
    throw error;
  }
};
