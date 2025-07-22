"use client";

import { useState } from 'react';
import { motion } from 'framer-motion';
import { Navbar } from '@/components/ui/navbar';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from '@/components/ui/card';
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from '@/components/ui/accordion';
import { TimeframeOption, timeframeOptions } from '@/lib/theme';
import Image from 'next/image';
import Link from 'next/link';

export default function HelpPage() {
  const [activeTab, setActiveTab] = useState('getting-started');

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-50 to-white">
      <Navbar />
      
      <main className="container mx-auto py-8 px-4 md:px-0">
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-purple-900 mb-2">Help & Documentation</h1>
          <p className="text-purple-700">Learn how to get the most accurate predictions from our AI-powered chart analysis</p>
        </div>
        
        <Tabs defaultValue={activeTab} onValueChange={setActiveTab} className="space-y-6">
          <TabsList className="bg-purple-100 p-1">
            <TabsTrigger 
              value="getting-started" 
              className="data-[state=active]:bg-purple-700 data-[state=active]:text-white"
            >
              Getting Started
            </TabsTrigger>
            <TabsTrigger 
              value="chart-requirements" 
              className="data-[state=active]:bg-purple-700 data-[state=active]:text-white"
            >
              Chart Requirements
            </TabsTrigger>
            <TabsTrigger 
              value="timeframes" 
              className="data-[state=active]:bg-purple-700 data-[state=active]:text-white"
            >
              Timeframes
            </TabsTrigger>
            <TabsTrigger 
              value="faq" 
              className="data-[state=active]:bg-purple-700 data-[state=active]:text-white"
            >
              FAQ
            </TabsTrigger>
          </TabsList>
          
          {/* Getting Started Tab */}
          <TabsContent value="getting-started">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5 }}
              >
                <Card className="h-full">
                  <CardHeader className="border-b border-purple-100">
                    <CardTitle className="text-purple-800">How to Use CryptoPredict</CardTitle>
                    <CardDescription className="text-purple-600">
                      Simple steps to get your first prediction
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="pt-6">
                    <ol className="space-y-6">
                      <li className="flex">
                        <div className="flex-shrink-0 flex h-8 w-8 rounded-full bg-purple-100 text-purple-800 items-center justify-center mr-3">
                          <span className="font-bold">1</span>
                        </div>
                        <div>
                          <h3 className="font-medium text-purple-900 mb-1">Screenshot Your Chart</h3>
                          <p className="text-sm text-purple-700">Take a clean screenshot of a candlestick chart from your preferred trading platform. Make sure the chart shows clear patterns and has good contrast.</p>
                        </div>
                      </li>
                      
                      <li className="flex">
                        <div className="flex-shrink-0 flex h-8 w-8 rounded-full bg-purple-100 text-purple-800 items-center justify-center mr-3">
                          <span className="font-bold">2</span>
                        </div>
                        <div>
                          <h3 className="font-medium text-purple-900 mb-1">Upload Your Chart</h3>
                          <p className="text-sm text-purple-700">Navigate to the Dashboard and upload your chart image. Supported formats are JPG and PNG.</p>
                        </div>
                      </li>
                      
                      <li className="flex">
                        <div className="flex-shrink-0 flex h-8 w-8 rounded-full bg-purple-100 text-purple-800 items-center justify-center mr-3">
                          <span className="font-bold">3</span>
                        </div>
                        <div>
                          <h3 className="font-medium text-purple-900 mb-1">Select Timeframe</h3>
                          <p className="text-sm text-purple-700">Choose the timeframe that matches your chart (15m, 1h, 4h, 1d, or 7d). This helps the AI understand the context of the patterns.</p>
                        </div>
                      </li>
                      
                      <li className="flex">
                        <div className="flex-shrink-0 flex h-8 w-8 rounded-full bg-purple-100 text-purple-800 items-center justify-center mr-3">
                          <span className="font-bold">4</span>
                        </div>
                        <div>
                          <h3 className="font-medium text-purple-900 mb-1">Analyze Chart</h3>
                          <p className="text-sm text-purple-700">Click the "Analyze Chart" button and wait a few seconds for our AI to process your image.</p>
                        </div>
                      </li>
                      
                      <li className="flex">
                        <div className="flex-shrink-0 flex h-8 w-8 rounded-full bg-purple-100 text-purple-800 items-center justify-center mr-3">
                          <span className="font-bold">5</span>
                        </div>
                        <div>
                          <h3 className="font-medium text-purple-900 mb-1">Review Results</h3>
                          <p className="text-sm text-purple-700">Examine the prediction (BUY, SELL, or HOLD), confidence score, detected patterns, and technical indicators.</p>
                        </div>
                      </li>
                    </ol>
                  </CardContent>
                  <CardFooter className="border-t border-purple-100 pt-4">
                    <Link href="/dashboard" className="text-purple-700 hover:text-purple-900 font-medium text-sm flex items-center">
                      <span>Go to Dashboard</span>
                      <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="ml-2">
                        <line x1="5" y1="12" x2="19" y2="12"></line>
                        <polyline points="12 5 19 12 12 19"></polyline>
                      </svg>
                    </Link>
                  </CardFooter>
                </Card>
              </motion.div>
              
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: 0.1 }}
              >
                <Card className="h-full">
                  <CardHeader className="border-b border-purple-100">
                    <CardTitle className="text-purple-800">Understanding Your Results</CardTitle>
                    <CardDescription className="text-purple-600">
                      How to interpret the AI predictions
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="pt-6 space-y-6">
                    <div>
                      <h3 className="font-medium text-purple-900 mb-2">Prediction Signal</h3>
                      <ul className="list-disc list-inside space-y-1 text-sm text-purple-700 pl-2">
                        <li><span className="text-purple-800 font-medium">BUY:</span> The AI detects bullish patterns suggesting an upward price movement</li>
                        <li><span className="text-red-500 font-medium">SELL:</span> The AI detects bearish patterns suggesting a downward price movement</li>
                        <li><span className="text-purple-400 font-medium">HOLD:</span> The AI detects consolidation or indecision patterns</li>
                      </ul>
                    </div>
                    
                    <div>
                      <h3 className="font-medium text-purple-900 mb-2">Confidence Score</h3>
                      <p className="text-sm text-purple-700">The confidence score (0-100%) indicates how certain the AI is about its prediction:</p>
                      <ul className="list-disc list-inside space-y-1 text-sm text-purple-700 pl-2 mt-1">
                        <li><span className="font-medium">90-100%:</span> Very high confidence</li>
                        <li><span className="font-medium">75-89%:</span> High confidence</li>
                        <li><span className="font-medium">60-74%:</span> Moderate confidence</li>
                        <li><span className="font-medium">Below 60%:</span> Low confidence - consider additional analysis</li>
                      </ul>
                    </div>
                    
                    <div>
                      <h3 className="font-medium text-purple-900 mb-2">Detected Patterns</h3>
                      <p className="text-sm text-purple-700">These are the specific candlestick patterns identified in your chart. Click on each pattern in the results page to learn more about what it indicates.</p>
                    </div>
                    
                    <div>
                      <h3 className="font-medium text-purple-900 mb-2">Technical Indicators</h3>
                      <p className="text-sm text-purple-700">The AI extracts or estimates key technical indicators from your chart image, such as RSI, MACD, and moving averages. These help inform the overall prediction.</p>
                    </div>
                  </CardContent>
                </Card>
              </motion.div>
            </div>
          </TabsContent>
          
          {/* Chart Requirements Tab */}
          <TabsContent value="chart-requirements">
            <div className="space-y-8">
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5 }}
              >
                <Card>
                  <CardHeader className="border-b border-purple-100">
                    <CardTitle className="text-purple-800">Optimal Chart Requirements</CardTitle>
                    <CardDescription className="text-purple-600">
                      For the most accurate predictions, your chart should meet these requirements
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="pt-6">
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                      <div className="space-y-4">
                        <div>
                          <h3 className="font-medium text-purple-900 mb-2">Chart Type & Quality</h3>
                          <ul className="list-disc list-inside space-y-1 text-sm text-purple-700 pl-2">
                            <li>Use <span className="font-medium">candlestick charts</span> (not line or bar charts)</li>
                            <li>High resolution images (at least 800x600 pixels)</li>
                            <li>Good contrast between background and candlesticks</li>
                            <li>Include at least 20-30 candles for pattern recognition</li>
                            <li>Avoid excessive indicators that may obscure candlesticks</li>
                          </ul>
                        </div>
                        
                        <div>
                          <h3 className="font-medium text-purple-900 mb-2">Recommended Indicators</h3>
                          <p className="text-sm text-purple-700">Including these indicators improves prediction accuracy:</p>
                          <ul className="list-disc list-inside space-y-1 text-sm text-purple-700 pl-2 mt-1">
                            <li>Volume indicator</li>
                            <li>Simple Moving Averages (SMA 20, 50, or 200)</li>
                            <li>Relative Strength Index (RSI)</li>
                            <li>MACD (Moving Average Convergence Divergence)</li>
                            <li>Bollinger Bands</li>
                          </ul>
                        </div>
                      </div>
                      
                      <div className="space-y-4">
                        <div>
                          <h3 className="font-medium text-purple-900 mb-2">Chart Timeframe</h3>
                          <p className="text-sm text-purple-700">Match your chart timeframe with your selection:</p>
                          <ul className="list-disc list-inside space-y-1 text-sm text-purple-700 pl-2 mt-1">
                            <li><span className="font-medium">15m:</span> Short-term intraday charts</li>
                            <li><span className="font-medium">1h:</span> Intraday trading charts</li>
                            <li><span className="font-medium">4h:</span> Short-term swing trading charts</li>
                            <li><span className="font-medium">1d:</span> Daily charts for medium-term analysis</li>
                            <li><span className="font-medium">7d:</span> Weekly charts for long-term analysis</li>
                          </ul>
                          <p className="text-xs text-purple-600 mt-2 italic">Note: The AI will match your selected timeframe with the chart - a mismatch can lead to incorrect predictions.</p>
                        </div>
                        
                        <div>
                          <h3 className="font-medium text-purple-900 mb-2">What to Avoid</h3>
                          <ul className="list-disc list-inside space-y-1 text-sm text-purple-700 pl-2">
                            <li>Charts with excessive drawings or annotations</li>
                            <li>Screenshots with trading platform UI elements</li>
                            <li>Charts with unusual color schemes</li>
                            <li>Charts showing very few candles</li>
                            <li>Heavily zoomed-in views that lack context</li>
                          </ul>
                        </div>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </motion.div>
              
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: 0.1 }}
              >
                <Card>
                  <CardHeader className="border-b border-purple-100">
                    <CardTitle className="text-purple-800">Chart Examples</CardTitle>
                    <CardDescription className="text-purple-600">
                      Examples of good and poor charts for analysis
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="pt-6">
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                      <div>
                        <h3 className="font-medium text-purple-900 mb-3">Good Chart Examples</h3>
                        <div className="bg-purple-50 p-4 rounded-lg border border-purple-200 text-center">
                          <div className="aspect-video bg-white flex items-center justify-center border border-purple-200 rounded-lg mb-2">
                            {/* Replace with actual good chart example */}
                            <svg xmlns="http://www.w3.org/2000/svg" width="80" height="80" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1" strokeLinecap="round" strokeLinejoin="round" className="text-purple-300">
                              <path d="M3 3v18h18" />
                              <path d="M7 12v5" />
                              <path d="M10 9v8" />
                              <path d="M13 7v10" />
                              <path d="M16 12v5" />
                              <path d="M19 8v9" />
                              <path d="M7 12v-2" />
                              <path d="M10 9v-3" />
                              <path d="M16 12v-2" />
                              <path d="M19 8v-2" />
                            </svg>
                          </div>
                          <ul className="list-disc list-inside text-left text-sm text-purple-700 space-y-1">
                            <li>Clear candlesticks with good contrast</li>
                            <li>Volume indicator visible</li>
                            <li>Key moving averages shown</li>
                            <li>Sufficient number of candles</li>
                            <li>No excessive drawings or annotations</li>
                          </ul>
                        </div>
                      </div>
                      
                      <div>
                        <h3 className="font-medium text-purple-900 mb-3">Poor Chart Examples</h3>
                        <div className="bg-purple-50 p-4 rounded-lg border border-purple-200 text-center">
                          <div className="aspect-video bg-white flex items-center justify-center border border-purple-200 rounded-lg mb-2">
                            {/* Replace with actual poor chart example */}
                            <svg xmlns="http://www.w3.org/2000/svg" width="80" height="80" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1" strokeLinecap="round" strokeLinejoin="round" className="text-red-300">
                              <path d="M3 3v18h18" />
                              <path d="M7 16v1" />
                              <path d="M10 14v3" />
                              <path d="M13 12v5" />
                              <path d="M16 10v7" />
                              <path d="M19 8v9" />
                              <line x1="5" y1="5" x2="19" y2="19" />
                              <line x1="19" y1="5" x2="5" y2="19" />
                            </svg>
                          </div>
                          <ul className="list-disc list-inside text-left text-sm text-purple-700 space-y-1">
                            <li>Too many indicators obscuring candles</li>
                            <li>Poor contrast or unusual colors</li>
                            <li>Too few candles to establish patterns</li>
                            <li>Excessive drawings or annotations</li>
                            <li>Trading platform UI elements visible</li>
                          </ul>
                        </div>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </motion.div>
            </div>
          </TabsContent>
          
          {/* Timeframes Tab */}
          <TabsContent value="timeframes">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5 }}
            >
              <Card>
                <CardHeader className="border-b border-purple-100">
                  <CardTitle className="text-purple-800">Choosing the Right Timeframe</CardTitle>
                  <CardDescription className="text-purple-600">
                    Different timeframes are suitable for different trading strategies
                  </CardDescription>
                </CardHeader>
                <CardContent className="pt-6">
                  <div className="space-y-8">
                    {[
                      {
                        timeframe: '15m',
                        title: '15 Minute Charts',
                        description: 'For intraday traders and scalpers looking for quick opportunities',
                        strengths: [
                          'Identifies short-term trading opportunities',
                          'Good for high-frequency day trading',
                          'Useful for entry and exit timing'
                        ],
                        limitations: [
                          'More prone to false signals and market noise',
                          'Requires constant monitoring',
                          'Generally lower prediction confidence'
                        ],
                        bestFor: 'Day traders and scalpers who actively monitor markets'
                      },
                      {
                        timeframe: '1h',
                        title: '1 Hour Charts',
                        description: 'Balanced timeframe for intraday and swing traders',
                        strengths: [
                          'Filters out some market noise compared to shorter timeframes',
                          'Still captures intraday price movements',
                          'Good balance of signal quality and trading frequency'
                        ],
                        limitations: [
                          'May miss very short-term opportunities',
                          'Still requires relatively frequent monitoring'
                        ],
                        bestFor: 'Active day traders and swing traders who check charts several times a day'
                      },
                      {
                        timeframe: '4h',
                        title: '4 Hour Charts',
                        description: 'Excellent for swing trading and medium-term positions',
                        strengths: [
                          'Filters out significant market noise',
                          'Captures medium-term trends more effectively',
                          'Higher confidence predictions than shorter timeframes'
                        ],
                        limitations: [
                          'Fewer trading signals than shorter timeframes',
                          'May be too slow for day traders'
                        ],
                        bestFor: 'Swing traders who hold positions for days or weeks'
                      },
                      {
                        timeframe: '1d',
                        title: 'Daily Charts',
                        description: 'For position traders focused on medium to long-term trends',
                        strengths: [
                          'Captures significant market trends',
                          'Provides high-quality, reliable signals',
                          'Requires less frequent monitoring'
                        ],
                        limitations: [
                          'May miss shorter-term opportunities',
                          'Less suitable for short-term trading'
                        ],
                        bestFor: 'Position traders who hold for weeks to months'
                      },
                      {
                        timeframe: '7d',
                        title: 'Weekly Charts',
                        description: 'For long-term investors and trend followers',
                        strengths: [
                          'Identifies major market trends',
                          'Filters out most market noise',
                          'Highest confidence predictions for long-term direction'
                        ],
                        limitations: [
                          'Very few trading signals',
                          'May be too slow for most active traders'
                        ],
                        bestFor: 'Long-term investors and macro trend followers'
                      }
                    ].map((item, index) => (
                      <div key={item.timeframe} className="border border-purple-200 rounded-lg p-6 bg-white">
                        <div className="flex items-center mb-4">
                          <div className="bg-purple-100 rounded-lg px-3 py-2 text-purple-800 font-bold text-lg mr-3">
                            {item.timeframe}
                          </div>
                          <h3 className="text-xl font-medium text-purple-900">{item.title}</h3>
                        </div>
                        
                        <p className="text-purple-700 mb-4">{item.description}</p>
                        
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                          <div>
                            <h4 className="font-medium text-purple-800 mb-2">Strengths</h4>
                            <ul className="list-disc list-inside text-sm text-purple-700 space-y-1">
                              {item.strengths.map((strength, i) => (
                                <li key={i}>{strength}</li>
                              ))}
                            </ul>
                          </div>
                          
                          <div>
                            <h4 className="font-medium text-purple-800 mb-2">Limitations</h4>
                            <ul className="list-disc list-inside text-sm text-purple-700 space-y-1">
                              {item.limitations.map((limitation, i) => (
                                <li key={i}>{limitation}</li>
                              ))}
                            </ul>
                          </div>
                        </div>
                        
                        <div className="mt-4 pt-3 border-t border-purple-100">
                          <span className="text-sm font-medium text-purple-800">Best for:</span>
                          <span className="text-sm text-purple-700 ml-2">{item.bestFor}</span>
                        </div>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            </motion.div>
          </TabsContent>
          
          {/* FAQ Tab */}
          <TabsContent value="faq">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5 }}
            >
              <Card>
                <CardHeader className="border-b border-purple-100">
                  <CardTitle className="text-purple-800">Frequently Asked Questions</CardTitle>
                  <CardDescription className="text-purple-600">
                    Common questions about our candlestick chart prediction tool
                  </CardDescription>
                </CardHeader>
                <CardContent className="pt-6">
                  <Accordion type="single" collapsible className="w-full">
                    <AccordionItem value="item-1" className="border-b border-purple-100">
                      <AccordionTrigger className="text-purple-800 hover:text-purple-700">
                        How accurate are the predictions?
                      </AccordionTrigger>
                      <AccordionContent className="text-purple-700">
                        <p>Our AI model has been trained on thousands of candlestick patterns and market data. Accuracy varies based on several factors:</p>
                        <ul className="list-disc list-inside text-sm mt-2 space-y-1">
                          <li>Chart quality and clarity</li>
                          <li>Timeframe appropriateness</li>
                          <li>Market conditions (trending vs. ranging)</li>
                          <li>Presence of strong technical patterns</li>
                        </ul>
                        <p className="mt-2">On average, prediction accuracy ranges from 65-85% when proper charts are used. The confidence score gives you an indication of the prediction's reliability.</p>
                      </AccordionContent>
                    </AccordionItem>
                    
                    <AccordionItem value="item-2" className="border-b border-purple-100">
                      <AccordionTrigger className="text-purple-800 hover:text-purple-700">
                        What charts and cryptocurrencies are supported?
                      </AccordionTrigger>
                      <AccordionContent className="text-purple-700">
                        <p>Our system supports candlestick charts from any trading platform and any cryptocurrency pair. The AI is trained on universal candlestick patterns that apply across all markets.</p>
                        <p className="mt-2">However, the prediction quality may vary based on the liquidity and volatility of the specific cryptocurrency. Major coins like BTC and ETH typically produce the most reliable signals due to their higher liquidity and more established patterns.</p>
                      </AccordionContent>
                    </AccordionItem>
                    
                    <AccordionItem value="item-3" className="border-b border-purple-100">
                      <AccordionTrigger className="text-purple-800 hover:text-purple-700">
                        Should I rely solely on these predictions for trading?
                      </AccordionTrigger>
                      <AccordionContent className="text-purple-700">
                        <p>No. While our AI provides valuable insights, it should be used as one tool in your trading toolkit, not as the sole basis for trading decisions.</p>
                        <p className="mt-2">We recommend combining our predictions with:</p>
                        <ul className="list-disc list-inside text-sm mt-2 space-y-1">
                          <li>Your own technical analysis</li>
                          <li>Fundamental analysis of the asset</li>
                          <li>Market sentiment and news</li>
                          <li>Proper risk management rules</li>
                        </ul>
                        <p className="mt-2 font-medium">Remember: All trading carries risk, and past performance is not indicative of future results.</p>
                      </AccordionContent>
                    </AccordionItem>
                    
                    <AccordionItem value="item-4" className="border-b border-purple-100">
                      <AccordionTrigger className="text-purple-800 hover:text-purple-700">
                        How does the system detect patterns and make predictions?
                      </AccordionTrigger>
                      <AccordionContent className="text-purple-700">
                        <p>Our system uses a multi-layered approach to analyze candlestick charts:</p>
                        <ol className="list-decimal list-inside text-sm mt-2 space-y-1">
                          <li><span className="font-medium">Image Processing:</span> Extracts candlestick data from your uploaded chart image</li>
                          <li><span className="font-medium">Pattern Recognition:</span> Identifies common candlestick patterns (engulfing, doji, hammer, etc.)</li>
                          <li><span className="font-medium">Technical Analysis:</span> Estimates technical indicators like RSI, MACD, and moving averages</li>
                          <li><span className="font-medium">Predictive Models:</span> Uses machine learning to evaluate all factors and generate a prediction</li>
                        </ol>
                        <p className="mt-2">The system also considers the timeframe you select to contextualize the patterns and adjust prediction confidence appropriately.</p>
                      </AccordionContent>
                    </AccordionItem>
                    
                    <AccordionItem value="item-5" className="border-b border-purple-100">
                      <AccordionTrigger className="text-purple-800 hover:text-purple-700">
                        Why do I get different predictions for the same chart?
                      </AccordionTrigger>
                      <AccordionContent className="text-purple-700">
                        <p>If you upload exactly the same chart image with the same timeframe selection, you should receive the same prediction. Our system caches results based on image content.</p>
                        <p className="mt-2">However, slight variations in the uploaded image (even screenshots of the same chart taken at different times) can result in different predictions. This could happen due to:</p>
                        <ul className="list-disc list-inside text-sm mt-2 space-y-1">
                          <li>Different image cropping or resolution</li>
                          <li>Changes in platform UI elements captured in the screenshot</li>
                          <li>Slight differences in chart scaling or zoom level</li>
                        </ul>
                        <p className="mt-2">For most consistent results, use the same exact image file when comparing predictions.</p>
                      </AccordionContent>
                    </AccordionItem>
                    
                    <AccordionItem value="item-6" className="border-b border-purple-100">
                      <AccordionTrigger className="text-purple-800 hover:text-purple-700">
                        Is my chart data stored or shared?
                      </AccordionTrigger>
                      <AccordionContent className="text-purple-700">
                        <p>Your uploaded chart images are temporarily stored on our servers only for the duration needed to process them. They are not permanently stored or shared with third parties.</p>
                        <p className="mt-2">We do keep a record of the prediction results in your local browser storage (not on our servers) so you can access your prediction history. This data stays on your device and is not accessible to others.</p>
                      </AccordionContent>
                    </AccordionItem>
                  </Accordion>
                </CardContent>
                <CardFooter className="border-t border-purple-100 flex justify-between">
                  <p className="text-sm text-purple-600">Can't find what you're looking for?</p>
                  <Link href="/education" className="text-sm text-purple-700 hover:text-purple-900 font-medium">
                    Visit the Education section
                  </Link>
                </CardFooter>
              </Card>
            </motion.div>
          </TabsContent>
        </Tabs>
      </main>
    </div>
  );
}
