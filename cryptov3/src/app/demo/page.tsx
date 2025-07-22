'use client';

import React, { useState, FormEvent } from 'react';
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import axios from 'axios';

// Toast implementation without dependency
const useToast = () => {
  return {
    toast: (props: { title: string; description: string; variant?: string }) => {
      // Simple fallback for the toast functionality
      alert(`${props.title}\n${props.description}`);
    }
  };
};

// Define TypeScript interfaces for our data
interface OHLCVData {
  timestamp: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

interface TechnicalIndicators {
  [key: string]: number | string;
}

interface PredictionResult {
  prediction: string;
  confidence: number;
  detected_patterns: string[];  // Array of strings
  technical_indicators: TechnicalIndicators;
  timestamp?: string;
}

export default function SimplifiedDemo() {
  const { toast } = useToast();
  const [loading, setLoading] = useState(false);
  const [timeframe, setTimeframe] = useState("1d");
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [selectedTab, setSelectedTab] = useState("ohlcv");
  
  // Generate sample OHLCV data
  const generateSampleData = (): OHLCVData[] => {
    const data: OHLCVData[] = [];
    const days = 30;
    let closePrice = 100;
    
    const today = new Date();
    
    for (let i = 0; i < days; i++) {
      const date = new Date();
      date.setDate(today.getDate() - (days - i));
      
      // Create a general trend with some randomness
      if (i > 0) {
        closePrice = closePrice * (1 + (Math.random() - 0.5) * 0.04);
      }
      
      const openPrice = closePrice * (1 + (Math.random() - 0.5) * 0.02);
      const highPrice = Math.max(openPrice, closePrice) * (1 + Math.random() * 0.01);
      const lowPrice = Math.min(openPrice, closePrice) * (1 - Math.random() * 0.01);
      const volume = 1000000 + (Math.random() - 0.5) * 400000;
      
      data.push({
        timestamp: date.toISOString().split('T')[0],
        open: parseFloat(openPrice.toFixed(2)),
        high: parseFloat(highPrice.toFixed(2)),
        low: parseFloat(lowPrice.toFixed(2)),
        close: parseFloat(closePrice.toFixed(2)),
        volume: Math.floor(volume)
      });
    }
    
    return data;
    
    // return data;
  };
  
  // Submit OHLCV data to API
  const handleOHLCVSubmit = async () => {
    setLoading(true);
    try {
      const ohlcvData = generateSampleData();
      
      // Use the specific OHLCV endpoint
      const response = await axios.post('http://localhost:8000/api/predict/ohlcv', {
        timeframe,
        chart_data: ohlcvData
      });
      
      setResult(response.data);
      toast({
        title: "Prediction Complete",
        description: `Prediction: ${response.data.prediction} with ${(response.data.confidence * 100).toFixed(2)}% confidence`,
      });
    } catch (error) {
      console.error(error);
      toast({
        title: "Error",
        description: "Failed to get prediction. Please try again.",
        variant: "destructive",
      });
    } finally {
      setLoading(false);
    }
  };
  
  // Upload and submit image to API
  const handleImageSubmit = async (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    
    const fileInput = document.getElementById('chart-image') as HTMLInputElement;
    if (!fileInput || !fileInput.files || !fileInput.files[0]) {
      toast({
        title: "No File Selected",
        description: "Please select an image file first.",
        variant: "destructive",
      });
      return;
    }
    
    setLoading(true);
    
    const formData = new FormData();
    formData.append('image', fileInput.files[0]);
    formData.append('timeframe', timeframe);
    
    try {
      const response = await axios.post('http://localhost:8000/api/predict', formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });
      
      setResult(response.data);
      toast({
        title: "Image Processed",
        description: `Prediction: ${response.data.prediction} with ${(response.data.confidence * 100).toFixed(2)}% confidence`,
      });
    } catch (error) {
      console.error(error);
      toast({
        title: "Error",
        description: "Failed to process image. Please try again.",
        variant: "destructive",
      });
    } finally {
      setLoading(false);
    }
  };
  
  return (
    <div className="container mx-auto py-8">
      <h1 className="text-3xl font-bold mb-6">Candlestick Chart Prediction Demo</h1>
      
      <Tabs 
        defaultValue="ohlcv" 
        className="w-full" 
        onValueChange={(value) => setSelectedTab(value)}
      >
        <TabsList className="grid w-full grid-cols-2">
          <TabsTrigger value="ohlcv">OHLCV Data</TabsTrigger>
          <TabsTrigger value="image">Image Upload</TabsTrigger>
        </TabsList>
        
        <TabsContent value="ohlcv" className="mt-4">
          <Card>
            <CardHeader>
              <CardTitle>OHLCV Data Prediction</CardTitle>
              <CardDescription>
                Submit OHLCV data for candlestick pattern analysis and prediction
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="mb-4">
                <label className="block text-sm font-medium mb-1">Timeframe</label>
                <Select value={timeframe} onValueChange={setTimeframe}>
                  <SelectTrigger className="w-full">
                    <SelectValue placeholder="Select timeframe" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="15m">15 Minutes</SelectItem>
                    <SelectItem value="1h">1 Hour</SelectItem>
                    <SelectItem value="4h">4 Hours</SelectItem>
                    <SelectItem value="1d">1 Day</SelectItem>
                    <SelectItem value="7d">1 Week</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div className="mb-4">
                <p className="text-sm text-gray-500">
                  Using randomly generated OHLCV data for demonstration
                </p>
              </div>
            </CardContent>
            <CardFooter>
              <Button 
                className="w-full" 
                onClick={handleOHLCVSubmit}
                disabled={loading}
              >
                {loading ? "Processing..." : "Run Prediction"}
              </Button>
            </CardFooter>
          </Card>
        </TabsContent>
        
        <TabsContent value="image" className="mt-4">
          <Card>
            <CardHeader>
              <CardTitle>Image-based Prediction</CardTitle>
              <CardDescription>
                Upload a candlestick chart image for pattern detection and prediction
              </CardDescription>
            </CardHeader>
            <CardContent>
              <form onSubmit={handleImageSubmit}>
                <div className="mb-4">
                  <label className="block text-sm font-medium mb-1">Timeframe</label>
                  <Select value={timeframe} onValueChange={setTimeframe}>
                    <SelectTrigger className="w-full">
                      <SelectValue placeholder="Select timeframe" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="15m">15 Minutes</SelectItem>
                      <SelectItem value="1h">1 Hour</SelectItem>
                      <SelectItem value="4h">4 Hours</SelectItem>
                      <SelectItem value="1d">1 Day</SelectItem>
                      <SelectItem value="7d">1 Week</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <div className="mb-4">
                  <label className="block text-sm font-medium mb-1">Chart Image</label>
                  <input
                    id="chart-image"
                    type="file"
                    accept="image/*"
                    className="w-full p-2 border rounded-md"
                  />
                </div>
                <Button 
                  type="submit" 
                  className="w-full" 
                  disabled={loading}
                >
                  {loading ? "Processing..." : "Upload & Analyze"}
                </Button>
              </form>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
      
      {/* Results section */}
      {result && (
        <Card className="mt-8">
          <CardHeader>
            <CardTitle>Prediction Results</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid gap-4 md:grid-cols-2">
              <div className="p-4 border rounded-lg">
                <h3 className="text-xl font-semibold mb-2">Prediction</h3>
                <div className="flex items-center space-x-2">
                  <span className={`text-2xl font-bold ${
                    result.prediction === "BUY" ? "text-green-500" : 
                    result.prediction === "SELL" ? "text-red-500" : "text-yellow-500"
                  }`}>
                    {result.prediction}
                  </span>
                  <span className="text-gray-500">
                    ({(result.confidence * 100).toFixed(2)}% confidence)
                  </span>
                </div>
              </div>
              
              <div className="p-4 border rounded-lg">
                <h3 className="text-xl font-semibold mb-2">Detected Patterns</h3>
                {result.detected_patterns && Array.isArray(result.detected_patterns) && result.detected_patterns.length > 0 ? (
                  <ul className="list-disc list-inside">
                    {result.detected_patterns.map((pattern, index) => (
                      <li key={index} className="capitalize">
                        {typeof pattern === 'string' ? pattern.replace(/_/g, ' ') : 'Unknown pattern'}
                      </li>
                    ))}
                  </ul>
                ) : (
                  <p className="text-gray-500">No specific patterns detected</p>
                )}
              </div>
              
              {result.technical_indicators && Object.keys(result.technical_indicators).length > 0 && (
                <div className="p-4 border rounded-lg md:col-span-2">
                  <h3 className="text-xl font-semibold mb-2">Technical Indicators</h3>
                  <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                    {Object.entries(result.technical_indicators).map(([key, value]) => (
                      <div key={key} className="p-2">
                        <div className="text-sm text-gray-600 capitalize">{key.replace(/_/g, ' ')}</div>
                        <div className="font-medium">
                          {typeof value === 'number' 
                            ? (value as number).toFixed(2) 
                            : String(value)
                          }
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
