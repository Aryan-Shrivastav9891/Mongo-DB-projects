"use client";

import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { PredictionResponse } from '@/lib/api-client';

export function PredictionDisplay() {
  const [predictionResult, setPredictionResult] = useState<PredictionResponse | null>(null);

  useEffect(() => {
    // Get the prediction result from localStorage
    const storedResult = localStorage.getItem('predictionResult');
    if (storedResult) {
      setPredictionResult(JSON.parse(storedResult));
    }
  }, []);

  if (!predictionResult) {
    return (
      <Card className="w-full border-purple-200 bg-gradient-to-br from-white to-purple-50">
        <CardHeader className="border-b border-purple-100">
          <CardTitle className="text-center text-purple-800">Prediction Result</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-center text-purple-400">No prediction data available</p>
        </CardContent>
      </Card>
    );
  }

  const { prediction, confidence } = predictionResult;
  const confidencePercentage = (confidence * 100).toFixed(1);

  // Determine the color based on prediction
  const getPredictionColor = () => {
    switch (prediction) {
      case 'BUY':
        return 'bg-purple-100 text-purple-700 border-purple-300';
      case 'SELL':
        return 'bg-red-100 text-red-700 border-red-200';
      case 'HOLD':
        return 'bg-purple-50 text-purple-600 border-purple-200';
      default:
        return 'bg-gray-100 text-gray-700 border-gray-200';
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.9 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.5 }}
    >
      <Card className="w-full border-purple-200 bg-gradient-to-br from-white to-purple-50">
        <CardHeader className="border-b border-purple-100">
          <CardTitle className="text-center text-purple-800">Prediction Result</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex flex-col items-center space-y-6">
            {/* Prediction signal */}
            <div className={`text-center p-6 rounded-full border-2 shadow-sm ${getPredictionColor()}`}>
              <span className="text-3xl font-bold">{prediction}</span>
            </div>
            
            {/* Confidence level */}
            <div className="w-full max-w-sm">
              <p className="text-sm text-purple-700 mb-2 text-center font-medium">Confidence Level</p>
              <div className="w-full bg-purple-100 rounded-full h-4">
                <div 
                  className={`h-4 rounded-full ${
                    confidence > 0.8 ? 'bg-purple-600' : 
                    confidence > 0.5 ? 'bg-purple-500' : 'bg-purple-400'
                  }`}
                  style={{ width: `${confidencePercentage}%` }}
                ></div>
              </div>
              <p className="text-right text-sm mt-1 text-purple-800 font-medium">{confidencePercentage}%</p>
            </div>
            
            {/* Additional info */}
            <div className="text-center">
              <p className="text-sm text-purple-700">
                Based on the provided chart and selected timeframe, our algorithm suggests a <strong className="text-purple-900">{prediction}</strong> action 
                with {confidencePercentage}% confidence.
              </p>
            </div>
          </div>
        </CardContent>
      </Card>
    </motion.div>
  );
}
