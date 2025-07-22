import { TimeframeOption } from './theme';

export interface PredictionResult {
  isChart: boolean;
  prediction?: string;
  confidence?: number;
  detected_patterns?: string[];
  visual_features?: string[];
  technical_indicators?: Record<string, number | string>;
  survey_prediction?: string;
  survey_confidence?: number;
  combined_prediction?: string;
  combined_confidence?: number;
  timestamp: string;
}

export async function analyzeCryptoImage(
  imageFile: File,
  timeframe: TimeframeOption,
  predictionInput?: string
): Promise<PredictionResult> {
  const formData = new FormData();
  formData.append('image', imageFile);
  formData.append('timeframe', timeframe);
  if (predictionInput) {
    formData.append('predictionInput', predictionInput);
  }
  
  const response = await fetch('/api/imgNewModel', {
    method: 'POST',
    body: formData,
  });
  
  if (!response.ok) {
    throw new Error(`Error: ${response.statusText}`);
  }
  
  return await response.json();
}

export function getPredictionColor(prediction: string): string {
  switch (prediction?.toUpperCase()) {
    case 'BUY':
    case 'UPTREND':
      return 'text-green-600';
    case 'SELL':
    case 'DOWNTREND':
      return 'text-red-600';
    case 'HOLD':
    case 'NEUTRAL':
      return 'text-yellow-600';
    default:
      return 'text-gray-600';
  }
}

export function getPredictionBgColor(prediction: string): string {
  switch (prediction?.toUpperCase()) {
    case 'BUY':
    case 'UPTREND':
      return 'bg-green-100 border-green-200';
    case 'SELL':
    case 'DOWNTREND':
      return 'bg-red-100 border-red-200';
    case 'HOLD':
    case 'NEUTRAL':
      return 'bg-yellow-100 border-yellow-200';
    default:
      return 'bg-gray-100 border-gray-200';
  }
}

export function formatTimestamp(timestamp: string): string {
  try {
    const date = new Date(timestamp);
    return date.toLocaleString();
  } catch (error) {
    return timestamp;
  }
}

export function getConfidenceLabel(confidence: number): string {
  if (confidence >= 0.85) return 'Very High';
  if (confidence >= 0.7) return 'High';
  if (confidence >= 0.55) return 'Moderate';
  if (confidence >= 0.4) return 'Low';
  return 'Very Low';
}
