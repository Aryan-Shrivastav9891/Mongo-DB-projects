"use client";

import { useState } from 'react';
import { motion } from 'framer-motion';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { TimeframeOption, timeframeOptions } from '@/lib/theme';

interface AnalysisResult {
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

export default function ImgNewModelPage() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [selectedTimeframe, setSelectedTimeframe] = useState<TimeframeOption>('1h');
  const [predictionInput, setPredictionInput] = useState<string>('');
  const [isLoading, setIsLoading] = useState(false);
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(null);
  const [activeTab, setActiveTab] = useState<string>('chart');

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setSelectedFile(file);
      
      // Create preview URL
      const reader = new FileReader();
      reader.onload = () => {
        setPreviewUrl(reader.result as string);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleTimeframeChange = (value: TimeframeOption) => {
    setSelectedTimeframe(value);
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!selectedFile) {
      alert('Please select an image file first');
      return;
    }
    
    setIsLoading(true);
    
    try {
      const formData = new FormData();
      formData.append('image', selectedFile);
      formData.append('timeframe', selectedTimeframe);
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
      
      const result = await response.json();
      setAnalysisResult(result);
      
      // Adjust active tab based on result type
      if (result.isChart) {
        setActiveTab('chart');
      } else {
        setActiveTab('general');
      }
    } catch (error) {
      console.error('Error analyzing image:', error);
      alert('Failed to analyze image. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  const renderPatternsList = () => {
    if (!analysisResult?.detected_patterns || analysisResult.detected_patterns.length === 0) {
      return <p className="text-gray-500">No patterns detected</p>;
    }

    return (
      <ul className="list-disc pl-5 space-y-1">
        {analysisResult.detected_patterns.map((pattern, idx) => (
          <li key={idx} className="text-gray-700">{pattern}</li>
        ))}
      </ul>
    );
  };

  const renderVisualFeatures = () => {
    if (!analysisResult?.visual_features || analysisResult.visual_features.length === 0) {
      return <p className="text-gray-500">No visual features detected</p>;
    }

    return (
      <ul className="list-disc pl-5 space-y-1">
        {analysisResult.visual_features.map((feature, idx) => (
          <li key={idx} className="text-gray-700">{feature}</li>
        ))}
      </ul>
    );
  };

  const renderTechnicalIndicators = () => {
    if (!analysisResult?.technical_indicators || Object.keys(analysisResult.technical_indicators).length === 0) {
      return <p className="text-gray-500">No technical indicators available</p>;
    }

    return (
      <div className="grid grid-cols-2 gap-2">
        {Object.entries(analysisResult.technical_indicators).map(([key, value]) => (
          <div key={key} className="flex justify-between">
            <span className="font-medium">{key}:</span>
            <span className="text-gray-700">{value}</span>
          </div>
        ))}
      </div>
    );
  };

  const getPredictionColor = (prediction: string) => {
    switch (prediction?.toUpperCase()) {
      case 'BUY':
        return 'bg-green-100 text-green-700 border-green-200';
      case 'SELL':
        return 'bg-red-100 text-red-700 border-red-200';
      case 'HOLD':
        return 'bg-yellow-100 text-yellow-700 border-yellow-200';
      case 'UPTREND':
        return 'bg-green-100 text-green-700 border-green-200';
      case 'DOWNTREND':
        return 'bg-red-100 text-red-700 border-red-200';
      case 'NEUTRAL':
        return 'bg-yellow-100 text-yellow-700 border-yellow-200';
      default:
        return 'bg-gray-100 text-gray-700 border-gray-200';
    }
  };

  return (
    <div className="min-h-screen bg-background">
      <header className="bg-secondary py-6">
        <div className="container mx-auto">
          <h1 className="text-3xl font-bold text-primary text-center">
            Advanced Image Analysis Model
          </h1>
          <p className="text-center text-gray-600 mt-2">
            Analyze images with our dual-algorithm approach for maximum prediction accuracy
          </p>
        </div>
      </header>

      <main className="container mx-auto py-8 px-4 md:px-0">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Upload Form */}
          <Card className="lg:col-span-1">
            <CardHeader>
              <CardTitle>Upload Image</CardTitle>
            </CardHeader>
            <CardContent>
              <form onSubmit={handleSubmit} className="space-y-6">
                <div className="space-y-2">
                  <label htmlFor="image-upload" className="block text-sm font-medium">
                    Select Image
                  </label>
                  <div className="border-2 border-dashed border-gray-300 rounded-lg p-4 text-center cursor-pointer hover:bg-gray-50 transition-colors">
                    <input
                      id="image-upload"
                      type="file"
                      accept="image/*"
                      onChange={handleFileChange}
                      className="hidden"
                    />
                    <div onClick={() => document.getElementById('image-upload')?.click()}>
                      {previewUrl ? (
                        <div className="space-y-2">
                          <img
                            src={previewUrl}
                            alt="Preview"
                            className="max-h-[200px] mx-auto rounded-md"
                          />
                          <p className="text-sm text-blue-600">Click to change image</p>
                        </div>
                      ) : (
                        <div className="space-y-2">
                          <div className="mx-auto w-12 h-12 flex items-center justify-center rounded-full bg-gray-100">
                            <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-gray-400">
                              <rect width="18" height="18" x="3" y="3" rx="2" ry="2" />
                              <circle cx="9" cy="9" r="2" />
                              <path d="m21 15-3.086-3.086a2 2 0 0 0-2.828 0L6 21" />
                            </svg>
                          </div>
                          <p className="text-sm text-gray-500">
                            Click to select or drag and drop
                          </p>
                        </div>
                      )}
                    </div>
                  </div>
                </div>

                <div className="space-y-2">
                  <label className="block text-sm font-medium">
                    Select Timeframe
                  </label>
                  <div className="flex flex-wrap gap-2">
                    {timeframeOptions.map((option) => (
                      <Button
                        key={option}
                        type="button"
                        onClick={() => handleTimeframeChange(option)}
                        variant={selectedTimeframe === option ? 'default' : 'outline'}
                        className="px-3 py-1 h-auto"
                      >
                        {option}
                      </Button>
                    ))}
                  </div>
                </div>

                <div className="space-y-2">
                  <label htmlFor="prediction-input" className="block text-sm font-medium">
                    Additional Context (Optional)
                  </label>
                  <textarea
                    id="prediction-input"
                    value={predictionInput}
                    onChange={(e) => setPredictionInput(e.target.value)}
                    placeholder="Add any additional context for the prediction..."
                    className="w-full border border-gray-300 rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-primary"
                    rows={3}
                  />
                </div>

                <Button 
                  type="submit" 
                  className="w-full"
                  disabled={!selectedFile || isLoading}
                >
                  {isLoading ? 'Analyzing...' : 'Analyze Image'}
                </Button>
              </form>
            </CardContent>
          </Card>

          {/* Results Section */}
          <div className="lg:col-span-2">
            {analysisResult ? (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5 }}
              >
                <Card>
                  <CardHeader>
                    <CardTitle>Analysis Results</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <Tabs defaultValue={activeTab} onValueChange={setActiveTab}>
                      <TabsList className="grid w-full grid-cols-2">
                        <TabsTrigger value="chart" disabled={!analysisResult.isChart}>
                          Chart Analysis
                        </TabsTrigger>
                        <TabsTrigger value="general">
                          {analysisResult.isChart ? 'General Features' : 'Image Features'}
                        </TabsTrigger>
                      </TabsList>
                      
                      {/* Chart Analysis Tab */}
                      <TabsContent value="chart" className="space-y-4">
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-4">
                          {/* Chart Pattern Analysis */}
                          <div className="space-y-2">
                            <h3 className="font-medium text-lg">Detected Patterns</h3>
                            {renderPatternsList()}
                          </div>

                          {/* Technical Indicators */}
                          <div className="space-y-2">
                            <h3 className="font-medium text-lg">Technical Indicators</h3>
                            {renderTechnicalIndicators()}
                          </div>
                        </div>

                        <div className="mt-6">
                          <h3 className="font-medium text-lg mb-3">Market Predictions</h3>
                          
                          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                            {/* Chart Algorithm Prediction */}
                            <div className="border rounded-lg p-4">
                              <h4 className="text-sm text-gray-500 mb-1">Chart Algorithm</h4>
                              <div className={`mt-2 px-3 py-1 rounded-full inline-block text-sm ${getPredictionColor(analysisResult.prediction || '')}`}>
                                {analysisResult.prediction || 'N/A'} 
                                {analysisResult.confidence !== undefined && 
                                  ` (${(analysisResult.confidence * 100).toFixed(1)}%)`}
                              </div>
                            </div>
                            
                            {/* Survey Algorithm Prediction */}
                            <div className="border rounded-lg p-4">
                              <h4 className="text-sm text-gray-500 mb-1">Survey Algorithm</h4>
                              <div className={`mt-2 px-3 py-1 rounded-full inline-block text-sm ${getPredictionColor(analysisResult.survey_prediction || '')}`}>
                                {analysisResult.survey_prediction || 'N/A'} 
                                {analysisResult.survey_confidence !== undefined && 
                                  ` (${(analysisResult.survey_confidence * 100).toFixed(1)}%)`}
                              </div>
                            </div>
                            
                            {/* Combined Prediction */}
                            <div className="border rounded-lg p-4 bg-gray-50">
                              <h4 className="text-sm text-gray-500 mb-1">Combined Prediction</h4>
                              <div className={`mt-2 px-3 py-1 rounded-full inline-block text-sm font-medium ${getPredictionColor(analysisResult.combined_prediction || '')}`}>
                                {analysisResult.combined_prediction || 'N/A'} 
                                {analysisResult.combined_confidence !== undefined && 
                                  ` (${(analysisResult.combined_confidence * 100).toFixed(1)}%)`}
                              </div>
                            </div>
                          </div>
                        </div>
                      </TabsContent>
                      
                      {/* General Features Tab */}
                      <TabsContent value="general" className="space-y-4">
                        <div className="mt-4">
                          <h3 className="font-medium text-lg mb-3">Visual Analysis</h3>
                          <div className="space-y-2">
                            <h4 className="font-medium">Detected Features</h4>
                            {renderVisualFeatures()}
                          </div>
                          
                          {!analysisResult.isChart && (
                            <div className="mt-6 p-4 bg-yellow-50 rounded-lg border border-yellow-200">
                              <h4 className="font-medium text-yellow-800">Non-Chart Image Detected</h4>
                              <p className="text-sm text-yellow-700 mt-1">
                                This image was not detected as a candlestick chart. Visual features have been extracted instead.
                              </p>
                            </div>
                          )}
                        </div>
                      </TabsContent>
                    </Tabs>
                  </CardContent>
                </Card>
              </motion.div>
            ) : (
              <div className="h-full flex items-center justify-center">
                <div className="text-center space-y-2 p-8">
                  <div className="mx-auto w-16 h-16 flex items-center justify-center rounded-full bg-primary/10">
                    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-primary">
                      <path d="M7 21h10M12 3v18m-5-7 5 5 5-5"/>
                    </svg>
                  </div>
                  <h3 className="font-medium text-lg">Upload an Image to Begin</h3>
                  <p className="text-gray-500 text-sm">
                    Our AI will analyze your image and provide detailed insights.
                  </p>
                </div>
              </div>
            )}
          </div>
        </div>
      </main>
    </div>
  );
}
