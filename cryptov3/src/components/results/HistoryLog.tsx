"use client";

import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { Card, CardContent, CardHeader, CardTitle, CardFooter } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { useRouter } from 'next/navigation';
import Link from 'next/link';

type HistoryItem = {
  id: string;
  timestamp: string;
  prediction: 'BUY' | 'SELL' | 'HOLD';
  confidence: number;
  timeframe: string;
};

export function HistoryLog() {
  const [historyItems, setHistoryItems] = useState<HistoryItem[]>([]);
  const router = useRouter();

  useEffect(() => {
    // Get current prediction result
    const storedResult = localStorage.getItem('predictionResult');
    
    if (storedResult) {
      const result = JSON.parse(storedResult);
      
      // Get existing history from localStorage
      const existingHistory = localStorage.getItem('predictionHistory');
      const history = existingHistory ? JSON.parse(existingHistory) : [];
      
      // Add new entry with timestamp and ensure unique ID
      const newEntry: HistoryItem = {
        id: `${Date.now()}-${Math.random().toString(36).substring(2, 9)}`,
        timestamp: result.timestamp || new Date().toISOString(),
        prediction: result.prediction,
        confidence: result.confidence,
        timeframe: result.timeframe || 'unknown',
      };
      
      // Update history (keep most recent 10 entries)
      const updatedHistory = [newEntry, ...history].slice(0, 10);
      
      // Save updated history
      localStorage.setItem('predictionHistory', JSON.stringify(updatedHistory));
      
      // Update state
      setHistoryItems(updatedHistory);
    } else {
      // Just load existing history if no new prediction
      const existingHistory = localStorage.getItem('predictionHistory');
      if (existingHistory) {
        // Ensure each item has a unique ID before setting state
        const history = JSON.parse(existingHistory);
        const historyWithUniqueIds = history.map((item: HistoryItem) => {
          // If an item doesn't have an ID or has a potentially duplicate ID (just timestamp),
          // generate a new unique ID
          if (!item.id || item.id.indexOf('-') === -1) {
            return {
              ...item,
              id: `${Date.now()}-${Math.random().toString(36).substring(2, 9)}`
            };
          }
          return item;
        });
        
        // Update localStorage with fixed IDs if needed
        if (JSON.stringify(history) !== JSON.stringify(historyWithUniqueIds)) {
          localStorage.setItem('predictionHistory', JSON.stringify(historyWithUniqueIds));
        }
        
        setHistoryItems(historyWithUniqueIds);
      }
    }
  }, []);

  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    return date.toLocaleString();
  };

  const getPredictionColor = (prediction: string) => {
    switch (prediction) {
      case 'BUY': return 'text-purple-600';
      case 'SELL': return 'text-red-500';
      case 'HOLD': return 'text-purple-400';
      default: return 'text-gray-600';
    }
  };

  const handleNewAnalysis = () => {
    router.push('/dashboard');
  };

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.9 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.5, delay: 0.2 }}
    >
      <Card className="w-full border-purple-200 bg-gradient-to-br from-white to-purple-50">
        <CardHeader className="border-b border-purple-100">
          <CardTitle className="text-center text-purple-800">Prediction History</CardTitle>
        </CardHeader>
        <CardContent>
          {historyItems.length > 0 ? (
            <div className="space-y-3 max-h-[300px] overflow-y-auto">
              {historyItems.map((item) => (
                <div key={item.id} className="border border-purple-200 rounded-lg p-3 hover:shadow-md hover:bg-purple-50 transition-all">
                  <div className="flex justify-between items-center">
                    <span className={`font-medium ${getPredictionColor(item.prediction)}`}>
                      {item.prediction}
                    </span>
                    <span className="text-sm text-purple-700">{formatDate(item.timestamp)}</span>
                  </div>
                  <div className="flex justify-between items-center mt-1 text-sm">
                    <span className="text-purple-900">Confidence: {(item.confidence * 100).toFixed(1)}%</span>
                    <span className="bg-purple-100 text-purple-800 px-2 py-0.5 rounded-full text-xs border border-purple-200">
                      {item.timeframe}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <p className="text-center text-purple-400 py-10">No prediction history available</p>
          )}
        </CardContent>
        <CardFooter className="border-t border-purple-100 flex flex-col space-y-3">
          <Button onClick={handleNewAnalysis} className="w-full bg-purple-600 hover:bg-purple-700 text-white">
            New Analysis
          </Button>
          <Link href="/history" className="w-full">
            <Button variant="outline" className="w-full border-purple-200 text-purple-700 hover:bg-purple-50">
              View Full History
            </Button>
          </Link>
        </CardFooter>
      </Card>
    </motion.div>
  );
}
