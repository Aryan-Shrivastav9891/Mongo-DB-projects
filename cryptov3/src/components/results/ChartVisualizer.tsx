"use client";

import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { PredictionResponse } from '@/lib/api-client';

export function ChartVisualizer() {
  const [chartImage, setChartImage] = useState<string | null>(null);
  const [patterns, setPatterns] = useState<string[]>([]);

  useEffect(() => {
    // Get the prediction result and uploaded image from localStorage
    const storedResult = localStorage.getItem('predictionResult');
    const storedImage = localStorage.getItem('chartImage');
    
    if (storedResult) {
      const result: PredictionResponse = JSON.parse(storedResult);
      setPatterns(result.detected_patterns || []);
    }
    
    if (storedImage) {
      setChartImage(storedImage);
    }
  }, []);

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.9 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.5, delay: 0.1 }}
    >
      <Card className="w-full border-purple-200 bg-gradient-to-br from-white to-purple-50">
        <CardHeader className="border-b border-purple-100">
          <CardTitle className="text-center text-purple-800">Chart Analysis</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-6">
            {chartImage ? (
              <div className="text-center">
                <img 
                  src={chartImage} 
                  alt="Analyzed chart" 
                  className="max-w-full max-h-[300px] mx-auto rounded-lg border border-purple-200"
                />
              </div>
            ) : (
              <div className="h-[200px] flex items-center justify-center bg-purple-50 rounded-lg border border-purple-100">
                <p className="text-purple-400">Chart image not available</p>
              </div>
            )}
            
            {/* Detected patterns */}
            <div>
              <h3 className="text-lg font-medium mb-2 text-purple-800">Detected Patterns</h3>
              {patterns && patterns.length > 0 ? (
                <div className="flex flex-wrap gap-2">
                  {patterns.map((pattern, index) => (
                    <span 
                      key={index}
                      className="px-3 py-1 rounded-full bg-purple-100 border border-purple-200 text-purple-700 text-sm font-medium"
                    >
                      {pattern}
                    </span>
                  ))}
                </div>
              ) : (
                <p className="text-sm text-purple-400">No specific patterns detected</p>
              )}
            </div>
          </div>
        </CardContent>
      </Card>
    </motion.div>
  );
}
