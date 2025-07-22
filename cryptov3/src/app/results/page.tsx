"use client";

import Link from 'next/link';
import { Button } from '@/components/ui/button';
import { PredictionDisplay } from '@/components/results/PredictionDisplay';
import { ChartVisualizer } from '@/components/results/ChartVisualizer';
import { HistoryLog } from '@/components/results/HistoryLog';
import { ApiStatus } from '@/components/ui/api-status';
import { Navbar } from '@/components/ui/navbar';

export default function ResultsPage() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-white to-purple-50">
      <Navbar />
      <header className="bg-purple-100 py-6 border-b border-purple-200">
        <div className="container mx-auto">
          <h1 className="text-3xl font-bold text-purple-800 text-center">
            Analysis Results
          </h1>
          <p className="text-center text-purple-600 mt-2">
            AI-generated predictions based on your candlestick chart
          </p>
        </div>
      </header>

      <main className="container mx-auto py-8 px-4 md:px-0">
        {/* Back to dashboard link */}
        <div className="mb-6">
          <Link href="/dashboard">
            <Button variant="outline" size="sm">
              <svg
                xmlns="http://www.w3.org/2000/svg"
                width="16"
                height="16"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
                className="mr-2"
              >
                <path d="M19 12H5" />
                <path d="M12 19l-7-7 7-7" />
              </svg>
              Back to Dashboard
            </Button>
          </Link>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
          {/* Prediction Display */}
          <div className="lg:col-span-1">
            <PredictionDisplay />
          </div>

          {/* Chart Visualizer */}
          <div className="lg:col-span-1">
            <ChartVisualizer />
          </div>

          {/* History Log */}
          <div className="lg:col-span-1">
            <HistoryLog />
          </div>
        </div>
      </main>
      
      <footer className="bg-purple-100 py-4 mt-8 border-t border-purple-200">
        <div className="container mx-auto px-4">
          <div className="flex flex-col md:flex-row justify-between items-center gap-4">
            <div className="text-sm text-purple-700">
              &copy; {new Date().getFullYear()} Candlestick Chart Prediction Platform. All rights reserved.
            </div>
            <ApiStatus apiUrl={process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'} />
          </div>
        </div>
      </footer>
    </div>
  );
}
