"use client";

import { useState } from 'react';
import { UploadCard } from '@/components/dashboard/UploadCard';
import { TimeframeSelector } from '@/components/dashboard/TimeframeSelector';
import { PreviewPane } from '@/components/dashboard/PreviewPane';
import { ApiStatus } from '@/components/ui/api-status';
import { Navbar } from '@/components/ui/navbar';
import { TimeframeOption } from '@/lib/theme';

export default function DashboardPage() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [selectedTimeframe, setSelectedTimeframe] = useState<TimeframeOption>('1h');

  const handleFileSelected = (file: File) => {
    setSelectedFile(file);
    
    // Store image in localStorage for use on results page
    const reader = new FileReader();
    reader.onload = () => {
      localStorage.setItem('chartImage', reader.result as string);
    };
    reader.readAsDataURL(file);
  };

  const handleTimeframeChange = (timeframe: TimeframeOption) => {
    setSelectedTimeframe(timeframe);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-50 to-white">
      <Navbar />
      <header className="bg-purple-900 py-6 shadow-md">
        <div className="container mx-auto">
          <h1 className="text-3xl font-bold text-white text-center">
            Candlestick Chart Prediction
          </h1>
          <p className="text-center text-purple-200 mt-2">
            Upload your chart, select a timeframe, and get AI-powered predictions
          </p>
        </div>
      </header>

      <main className="container mx-auto py-8 px-4 md:px-0">
        {/* Quick Stats Section */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
          <div className="bg-white rounded-lg shadow-md border border-purple-200 p-6 flex items-center">
            <div className="rounded-full bg-purple-100 p-3 mr-4">
              <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-purple-800">
                <path d="M12 2v20M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6" />
              </svg>
            </div>
            <div>
              <p className="text-sm text-purple-600">Market Trends</p>
              <h3 className="font-bold text-2xl text-purple-900">BTC +2.4%</h3>
            </div>
          </div>
          <div className="bg-white rounded-lg shadow-md border border-purple-200 p-6 flex items-center">
            <div className="rounded-full bg-purple-100 p-3 mr-4">
              <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-purple-800">
                <path d="m22 2-7 20-4-9-9-4Z" />
                <path d="M22 2 11 13" />
              </svg>
            </div>
            <div>
              <p className="text-sm text-purple-600">Prediction Accuracy</p>
              <h3 className="font-bold text-2xl text-purple-900">87.5%</h3>
            </div>
          </div>
          <div className="bg-white rounded-lg shadow-md border border-purple-200 p-6 flex items-center">
            <div className="rounded-full bg-purple-100 p-3 mr-4">
              <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-purple-800">
                <path d="M3 3v18h18" />
                <path d="m19 9-5 5-4-4-3 3" />
              </svg>
            </div>
            <div>
              <p className="text-sm text-purple-600">Recent Signals</p>
              <h3 className="font-bold text-2xl text-purple-900">
                <span className="text-purple-700">BUY</span> (4) <span className="text-red-500">SELL</span> (2)
              </h3>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
          {/* Upload Card */}
          <div className="lg:col-span-1">
            <UploadCard onFileSelected={handleFileSelected} />
          </div>

          {/* Timeframe Selector */}
          <div className="lg:col-span-1">
            <TimeframeSelector
              selectedTimeframe={selectedTimeframe}
              onTimeframeChange={handleTimeframeChange}
            />
          </div>

          {/* Preview Pane */}
          <div className="lg:col-span-1">
            <PreviewPane
              selectedFile={selectedFile}
              selectedTimeframe={selectedTimeframe}
            />
          </div>
        </div>
        
        {/* Market Insights Section */}
        <div className="mt-10">
          <h2 className="text-2xl font-bold mb-4 text-purple-900">Market Insights</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="bg-white rounded-lg shadow-md border border-purple-200 p-6">
              <h3 className="font-semibold text-lg mb-3 text-purple-800">Pattern Frequency (Last 7 Days)</h3>
              <div className="space-y-2">
                <div className="flex justify-between items-center">
                  <span className="text-purple-700">Bullish Engulfing</span>
                  <div className="w-2/3 bg-purple-100 rounded-full h-3">
                    <div className="bg-purple-600 h-3 rounded-full" style={{ width: '65%' }}></div>
                  </div>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-purple-700">Bearish Engulfing</span>
                  <div className="w-2/3 bg-purple-100 rounded-full h-3">
                    <div className="bg-red-500 h-3 rounded-full" style={{ width: '40%' }}></div>
                  </div>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-purple-700">Doji</span>
                  <div className="w-2/3 bg-purple-100 rounded-full h-3">
                    <div className="bg-purple-400 h-3 rounded-full" style={{ width: '30%' }}></div>
                  </div>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-purple-700">Hammer</span>
                  <div className="w-2/3 bg-purple-100 rounded-full h-3">
                    <div className="bg-purple-600 h-3 rounded-full" style={{ width: '25%' }}></div>
                  </div>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-purple-700">Shooting Star</span>
                  <div className="w-2/3 bg-purple-100 rounded-full h-3">
                    <div className="bg-red-500 h-3 rounded-full" style={{ width: '20%' }}></div>
                  </div>
                </div>
              </div>
            </div>
            
            <div className="bg-white rounded-lg shadow-md border border-purple-200 p-6">
              <h3 className="font-semibold text-lg mb-3 text-purple-800">Recent Predictions</h3>
              <div className="space-y-4">
                <div className="flex justify-between items-center pb-2 border-b border-purple-100">
                  <span className="text-sm text-purple-700">BTC/USD (1h)</span>
                  <div className="flex items-center">
                    <span className="font-medium text-purple-700 mr-2">BUY</span>
                    <span className="text-xs text-purple-500">85% conf.</span>
                  </div>
                </div>
                <div className="flex justify-between items-center pb-2 border-b border-purple-100">
                  <span className="text-sm text-purple-700">ETH/USD (4h)</span>
                  <div className="flex items-center">
                    <span className="font-medium text-red-500 mr-2">SELL</span>
                    <span className="text-xs text-purple-500">78% conf.</span>
                  </div>
                </div>
                <div className="flex justify-between items-center pb-2 border-b border-purple-100">
                  <span className="text-sm text-purple-700">XRP/USD (1d)</span>
                  <div className="flex items-center">
                    <span className="font-medium text-purple-400 mr-2">HOLD</span>
                    <span className="text-xs text-purple-500">62% conf.</span>
                  </div>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-sm text-purple-700">SOL/USD (1h)</span>
                  <div className="flex items-center">
                    <span className="font-medium text-purple-700 mr-2">BUY</span>
                    <span className="text-xs text-purple-500">91% conf.</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
        
        {/* Technical Analysis Section */}
        <div className="mt-10">
          <h2 className="text-2xl font-bold mb-4 text-purple-900">Popular Technical Indicators</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            {['RSI', 'MACD', 'Bollinger Bands', 'Moving Averages'].map((indicator) => (
              <div key={indicator} className="bg-white rounded-lg shadow-md border border-purple-200 p-4 hover:shadow-lg transition-shadow cursor-pointer hover:bg-purple-50">
                <h3 className="font-medium text-center text-purple-800">{indicator}</h3>
                <p className="text-xs text-center text-purple-500 mt-1">Click to learn more</p>
              </div>
            ))}
          </div>
        </div>
      </main>
      
      <footer className="bg-purple-900 py-4 mt-8 shadow-inner">
        <div className="container mx-auto px-4">
          <div className="flex flex-col md:flex-row justify-between items-center gap-4">
            <div className="text-sm text-purple-200">
              &copy; {new Date().getFullYear()} Candlestick Chart Prediction Platform. All rights reserved.
            </div>
            <ApiStatus apiUrl={process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'} />
          </div>
        </div>
      </footer>
    </div>
  );
}
