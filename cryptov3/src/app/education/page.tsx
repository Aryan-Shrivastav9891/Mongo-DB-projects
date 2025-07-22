"use client";

import { useState } from 'react';
import { motion } from 'framer-motion';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Navbar } from '@/components/ui/navbar';

interface PatternInfo {
  name: string;
  description: string;
  significance: string;
  image: string;
}

export default function EducationPage() {
  const [activeTab, setActiveTab] = useState('patterns');
  
  const candlestickPatterns: PatternInfo[] = [
    {
      name: 'Bullish Engulfing',
      description: 'A bullish reversal pattern that forms after a downtrend. It consists of a small bearish candle followed by a larger bullish candle that completely engulfs the previous candle.',
      significance: 'Indicates strong buying pressure and potential trend reversal from bearish to bullish.',
      image: '/patterns/bullish-engulfing.png'
    },
    {
      name: 'Bearish Engulfing',
      description: 'A bearish reversal pattern that forms after an uptrend. It consists of a small bullish candle followed by a larger bearish candle that completely engulfs the previous candle.',
      significance: 'Indicates strong selling pressure and potential trend reversal from bullish to bearish.',
      image: '/patterns/bearish-engulfing.png'
    },
    {
      name: 'Doji',
      description: 'A candlestick with a very small body (open and close prices are very close) and significant upper and lower shadows.',
      significance: 'Represents indecision in the market and potential trend reversal, especially after a strong trend.',
      image: '/patterns/doji.png'
    },
    {
      name: 'Hammer',
      description: 'A bullish reversal pattern with a small body at the upper end of the trading range and a long lower shadow (at least twice the length of the body).',
      significance: 'Indicates potential bullish reversal after a downtrend.',
      image: '/patterns/hammer.png'
    },
    {
      name: 'Shooting Star',
      description: 'A bearish reversal pattern with a small body at the lower end of the trading range and a long upper shadow.',
      significance: 'Indicates potential bearish reversal after an uptrend.',
      image: '/patterns/shooting-star.png'
    },
  ];

  const technicalIndicators = [
    {
      name: 'RSI (Relative Strength Index)',
      description: 'A momentum indicator that measures the magnitude of recent price changes to evaluate overbought or oversold conditions.',
      interpretation: 'Values above 70 typically indicate overbought conditions, while values below 30 suggest oversold conditions.'
    },
    {
      name: 'MACD (Moving Average Convergence Divergence)',
      description: 'A trend-following momentum indicator showing the relationship between two moving averages of a security price.',
      interpretation: 'MACD crossing above the signal line is bullish, while crossing below is bearish. Divergence between MACD and price can signal potential reversals.'
    },
    {
      name: 'Bollinger Bands',
      description: 'A volatility indicator consisting of a middle band (SMA) and two outer bands that are standard deviations away from the middle.',
      interpretation: 'Price touching the upper band may indicate overbought conditions, while touching the lower band may indicate oversold conditions. Band contraction suggests decreasing volatility, while expansion suggests increasing volatility.'
    },
    {
      name: 'Moving Averages',
      description: 'A calculation used to analyze data points by creating a series of averages of different subsets of the full data set.',
      interpretation: 'Short-term MA crossing above long-term MA is a bullish signal (golden cross), while short-term MA crossing below long-term MA is a bearish signal (death cross).'
    },
  ];

  const tradingStrategies = [
    {
      name: 'Trend Following',
      description: 'A strategy that aims to capitalize on the momentum of existing market trends.',
      steps: [
        'Identify the current market trend using moving averages or trendlines.',
        'Enter trades in the direction of the trend when price pulls back.',
        'Set stop-loss orders to protect against unexpected reversals.',
        'Hold positions as long as the trend continues and exit when trend shows signs of reversal.'
      ]
    },
    {
      name: 'Mean Reversion',
      description: 'A strategy based on the idea that prices and returns eventually move back toward their historical average or mean.',
      steps: [
        'Identify when a security has deviated significantly from its historical average.',
        'Enter trades in the opposite direction of the deviation, expecting a reversion to the mean.',
        'Use indicators like RSI or Bollinger Bands to identify overbought or oversold conditions.',
        'Set profit targets based on historical means and strict stop-loss levels.'
      ]
    },
    {
      name: 'Breakout Trading',
      description: 'A strategy that involves entering a trade when the price breaks through an established support or resistance level.',
      steps: [
        'Identify key support and resistance levels or chart patterns.',
        'Enter a trade when price breaks through these levels on increased volume.',
        'Set stop-loss orders below support (for long trades) or above resistance (for short trades).',
        'Target profits at the next significant support/resistance level or based on measured moves.'
      ]
    },
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-50 to-white">
      <Navbar />
      
      <main className="container mx-auto py-8 px-4 md:px-0">
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-purple-900 mb-2">Trading Education Center</h1>
          <p className="text-purple-700">Learn about candlestick patterns, technical indicators, and trading strategies</p>
        </div>
        
        <Tabs defaultValue={activeTab} onValueChange={setActiveTab} className="space-y-6">
          <TabsList className="bg-purple-100 p-1">
            <TabsTrigger 
              value="patterns" 
              className="data-[state=active]:bg-purple-700 data-[state=active]:text-white"
            >
              Candlestick Patterns
            </TabsTrigger>
            <TabsTrigger 
              value="indicators" 
              className="data-[state=active]:bg-purple-700 data-[state=active]:text-white"
            >
              Technical Indicators
            </TabsTrigger>
            <TabsTrigger 
              value="strategies" 
              className="data-[state=active]:bg-purple-700 data-[state=active]:text-white"
            >
              Trading Strategies
            </TabsTrigger>
          </TabsList>
          
          <TabsContent value="patterns" className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {candlestickPatterns.map((pattern, index) => (
                <motion.div 
                  key={pattern.name}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.3, delay: index * 0.1 }}
                >
                  <Card className="h-full border-purple-200 bg-white shadow-sm hover:shadow-md transition-all">
                    <CardHeader className="border-b border-purple-100">
                      <CardTitle className="text-center text-purple-800">{pattern.name}</CardTitle>
                    </CardHeader>
                    <CardContent className="pt-4">
                      <div className="flex justify-center mb-4">
                        <div className="w-24 h-24 bg-purple-100 rounded-md flex items-center justify-center border border-purple-200">
                          {/* Replace with actual images once available */}
                          <svg xmlns="http://www.w3.org/2000/svg" width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-purple-700">
                            <path d="M8 22h8" />
                            <path d="M12 11v11" />
                            <path d="m19 3-7 8-7-8Z" />
                          </svg>
                        </div>
                      </div>
                      <div>
                        <h3 className="font-medium text-purple-900 mb-2">Description</h3>
                        <p className="text-sm text-purple-700 mb-3">{pattern.description}</p>
                        
                        <h3 className="font-medium text-purple-900 mb-2">Significance</h3>
                        <p className="text-sm text-purple-700">{pattern.significance}</p>
                      </div>
                    </CardContent>
                  </Card>
                </motion.div>
              ))}
            </div>
          </TabsContent>
          
          <TabsContent value="indicators" className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {technicalIndicators.map((indicator, index) => (
                <motion.div 
                  key={indicator.name}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.3, delay: index * 0.1 }}
                >
                  <Card className="h-full border-purple-200 bg-white shadow-sm hover:shadow-md transition-all">
                    <CardHeader className="border-b border-purple-100">
                      <CardTitle className="text-purple-800">{indicator.name}</CardTitle>
                    </CardHeader>
                    <CardContent className="pt-4 space-y-4">
                      <div>
                        <h3 className="font-medium text-purple-900 mb-1">What it is</h3>
                        <p className="text-sm text-purple-700">{indicator.description}</p>
                      </div>
                      <div>
                        <h3 className="font-medium text-purple-900 mb-1">How to interpret</h3>
                        <p className="text-sm text-purple-700">{indicator.interpretation}</p>
                      </div>
                    </CardContent>
                  </Card>
                </motion.div>
              ))}
            </div>
          </TabsContent>
          
          <TabsContent value="strategies" className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              {tradingStrategies.map((strategy, index) => (
                <motion.div 
                  key={strategy.name}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.3, delay: index * 0.1 }}
                >
                  <Card className="h-full border-purple-200 bg-white shadow-sm hover:shadow-md transition-all">
                    <CardHeader className="border-b border-purple-100">
                      <CardTitle className="text-purple-800">{strategy.name}</CardTitle>
                      <CardDescription className="text-purple-600">{strategy.description}</CardDescription>
                    </CardHeader>
                    <CardContent className="pt-4">
                      <h3 className="font-medium text-purple-900 mb-2">Implementation Steps</h3>
                      <ol className="list-decimal list-inside space-y-2">
                        {strategy.steps.map((step, i) => (
                          <li key={i} className="text-sm text-purple-700">{step}</li>
                        ))}
                      </ol>
                    </CardContent>
                  </Card>
                </motion.div>
              ))}
            </div>
          </TabsContent>
        </Tabs>
      </main>
    </div>
  );
}
