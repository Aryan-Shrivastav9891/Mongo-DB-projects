"use client";

import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { Card, CardContent, CardHeader, CardTitle, CardFooter } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { TimeframeOption } from '@/lib/theme';
import { toast } from 'sonner';
import { useRouter } from 'next/navigation';
import { predictCandlestickChart } from '@/lib/api-client';

interface PreviewPaneProps {
  selectedFile: File | null;
  selectedTimeframe: TimeframeOption;
}

export function PreviewPane({ selectedFile, selectedTimeframe }: PreviewPaneProps) {
  const [isLoading, setIsLoading] = useState(false);
  const router = useRouter();

  const handleAnalyze = async () => {
    if (!selectedFile) {
      toast.error('Please upload a candlestick chart image first');
      return;
    }

    setIsLoading(true);
    try {
      // Call the API to analyze the chart
      const result = await predictCandlestickChart(selectedFile, selectedTimeframe);
      
      // Store the result in localStorage for the results page to use
      localStorage.setItem('predictionResult', JSON.stringify(result));
      
      // Navigate to results page
      router.push('/results');
    } catch (error) {
      console.error('Error analyzing chart:', error);
      toast.error('Failed to analyze the chart. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay: 0.2 }}
    >
      <Card className="w-full border-purple-200 bg-gradient-to-br from-white to-purple-50 shadow-md">
        <CardHeader className="border-b border-purple-100">
          <CardTitle className="text-center text-purple-800">Analysis Preview</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div className="rounded-lg border border-purple-200 p-4 bg-white">
              <h3 className="font-medium mb-2 text-purple-800">Selected Chart</h3>
              {selectedFile ? (
                <div className="text-center">
                  <img 
                    src={URL.createObjectURL(selectedFile)} 
                    alt="Selected chart" 
                    className="max-h-[200px] mx-auto mb-2 border border-purple-200 rounded-md"
                  />
                  <p className="text-sm text-purple-600">{selectedFile.name}</p>
                </div>
              ) : (
                <p className="text-sm text-purple-400 text-center py-10">
                  No chart selected yet. Please upload an image.
                </p>
              )}
            </div>
            
            <div className="rounded-lg border border-purple-200 p-4 bg-white">
              <h3 className="font-medium mb-2 text-purple-800">Selected Timeframe</h3>
              <div className="text-center">
                <div className="inline-block px-3 py-1 rounded-full bg-purple-100 text-purple-800 font-medium border border-purple-200">
                  {selectedTimeframe}
                </div>
              </div>
            </div>
            
            <div className="rounded-lg border border-purple-200 p-4 bg-purple-50">
              <h3 className="font-medium mb-2 text-purple-800">Expected Analysis</h3>
              <div className="text-sm space-y-2 text-purple-700">
                <div className="flex justify-between">
                  <span>Candlestick Pattern Detection</span>
                  <span>✓</span>
                </div>
                <div className="flex justify-between">
                  <span>Technical Indicators</span>
                  <span>✓</span>
                </div>
                <div className="flex justify-between">
                  <span>Price Action Analysis</span>
                  <span>✓</span>
                </div>
                <div className="flex justify-between">
                  <span>Support/Resistance Levels</span>
                  <span>✓</span>
                </div>
                <div className="flex justify-between">
                  <span>Trend Direction</span>
                  <span>✓</span>
                </div>
                <div className="flex justify-between font-medium pt-2 border-t border-purple-200 mt-2 text-purple-900">
                  <span>Buy/Sell Signal</span>
                  <span>✓</span>
                </div>
              </div>
            </div>
            
            <div className="rounded-lg border border-purple-200 p-4 bg-purple-900/10">
              <h3 className="font-medium mb-2 text-purple-800">Pro Tips</h3>
              <ul className="text-sm text-purple-700 space-y-1 list-disc pl-4">
                <li>Use clear, high-resolution chart images</li>
                <li>Ensure price action is visible in the image</li>
                <li>Include volume indicators when possible</li>
                <li>Select the correct timeframe for best results</li>
              </ul>
            </div>
          </div>
        </CardContent>
        <CardFooter className="flex flex-col gap-4 border-t border-purple-100">
          <Button 
            onClick={handleAnalyze}
            disabled={!selectedFile || isLoading}
            className="w-full bg-purple-700 hover:bg-purple-800 text-white"
            size="lg"
          >
            {isLoading ? (
              <div className="flex items-center">
                <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
                Analyzing...
              </div>
            ) : 'Analyze Chart'}
          </Button>
          <p className="text-xs text-center text-purple-600">Analysis typically takes 2-3 seconds</p>
        </CardFooter>
      </Card>
    </motion.div>
  );
}
