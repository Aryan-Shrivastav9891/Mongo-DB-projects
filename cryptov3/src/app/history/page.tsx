"use client";

import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { useRouter } from 'next/navigation';
import { Navbar } from '@/components/ui/navbar';

type HistoryItem = {
  id: string;
  timestamp: string;
  prediction: 'BUY' | 'SELL' | 'HOLD';
  confidence: number;
  timeframe: string;
};

export default function HistoryPage() {
  const [historyItems, setHistoryItems] = useState<HistoryItem[]>([]);
  const [showAll, setShowAll] = useState(false);
  const router = useRouter();

  useEffect(() => {
    // Load existing history
    const existingHistory = localStorage.getItem('predictionHistory');
    if (existingHistory) {
      const history = JSON.parse(existingHistory);
      // Display all history items on this dedicated page
      setHistoryItems(history);
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

  const clearHistory = () => {
    if (confirm('Are you sure you want to clear all prediction history?')) {
      localStorage.removeItem('predictionHistory');
      setHistoryItems([]);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-50 to-white">
      <Navbar />
      
      <main className="container mx-auto py-8 px-4 md:px-0">
        <div className="flex justify-between items-center mb-6">
          <h1 className="text-2xl font-bold text-purple-900">Complete Prediction History</h1>
          {historyItems.length > 0 && (
            <Button 
              variant="outline" 
              className="border-red-300 text-red-600 hover:bg-red-50"
              onClick={clearHistory}
            >
              Clear History
            </Button>
          )}
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {historyItems.length > 0 ? (
            historyItems.map((item) => (
              <motion.div
                key={item.id}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.3 }}
                className="border border-purple-200 rounded-lg p-4 bg-white shadow-sm hover:shadow-md transition-all"
              >
                <div className="flex justify-between items-center mb-3">
                  <span className={`text-lg font-medium ${getPredictionColor(item.prediction)}`}>
                    {item.prediction}
                  </span>
                  <span className="bg-purple-100 text-purple-800 px-2 py-0.5 rounded-full text-xs border border-purple-200">
                    {item.timeframe}
                  </span>
                </div>
                
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <span className="text-purple-700">Confidence:</span>
                    <span className="font-medium text-purple-900">{(item.confidence * 100).toFixed(1)}%</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-purple-700">Date:</span>
                    <span className="text-sm">{formatDate(item.timestamp)}</span>
                  </div>
                  
                  <div className="w-full bg-purple-100 rounded-full h-2 mt-2">
                    <div 
                      className={`h-2 rounded-full ${
                        item.confidence > 0.8 ? 'bg-purple-600' : 
                        item.confidence > 0.5 ? 'bg-purple-500' : 'bg-purple-400'
                      }`}
                      style={{ width: `${(item.confidence * 100)}%` }}
                    ></div>
                  </div>
                </div>
              </motion.div>
            ))
          ) : (
            <div className="col-span-full">
              <Card className="w-full border-purple-200 bg-gradient-to-br from-white to-purple-50">
                <CardContent className="flex flex-col items-center justify-center py-10">
                  <svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" className="text-purple-300 mb-4">
                    <circle cx="12" cy="12" r="10" />
                    <polyline points="12 6 12 12 16 14" />
                  </svg>
                  <h2 className="text-xl font-medium text-purple-800 mb-2">No Prediction History</h2>
                  <p className="text-purple-600 mb-6 text-center">You haven't made any predictions yet. Start by analyzing a candlestick chart.</p>
                  <Button 
                    onClick={() => router.push('/dashboard')} 
                    className="bg-purple-700 hover:bg-purple-800 text-white"
                  >
                    Go to Dashboard
                  </Button>
                </CardContent>
              </Card>
            </div>
          )}
        </div>
      </main>
    </div>
  );
}
