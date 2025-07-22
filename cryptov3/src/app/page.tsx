"use client";

import { motion } from 'framer-motion';
import { Button } from '@/components/ui/button';
import Link from 'next/link';
import { Navbar } from '@/components/ui/navbar';

export default function HomePage() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-50 to-white">
      <Navbar />
      
      <div className="container mx-auto px-4 py-16 md:py-24">
        <div className="flex flex-col md:flex-row items-center">
          <div className="md:w-1/2 mb-10 md:mb-0">
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.5 }}
            >
              <h1 className="text-4xl md:text-5xl font-bold text-purple-900 mb-4">
                AI-Powered Candlestick Chart Analysis
              </h1>
              <p className="text-lg text-purple-700 mb-8 max-w-xl">
                Upload your crypto chart, select a timeframe, and get accurate predictions powered by advanced machine learning algorithms.
              </p>
              <div className="flex flex-col sm:flex-row gap-4">
                <Link href="/dashboard">
                  <Button className="bg-purple-700 hover:bg-purple-800 text-white px-6 py-6 text-lg">
                    Get Started
                  </Button>
                </Link>
                <Link href="/education">
                  <Button variant="outline" className="border-purple-300 text-purple-700 hover:bg-purple-50 px-6 py-6 text-lg">
                    Learn More
                  </Button>
                </Link>
              </div>
            </motion.div>
          </div>
          
          <div className="md:w-1/2">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.2 }}
              className="bg-white p-6 rounded-xl shadow-xl border border-purple-200"
            >
              <div className="aspect-video bg-purple-50 rounded-lg flex items-center justify-center mb-4">
                {/* Placeholder for chart image or illustration */}
                <svg xmlns="http://www.w3.org/2000/svg" width="120" height="120" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" className="text-purple-300">
                  <path d="M3 3v18h18" />
                  <path d="M7 17l3-4 4 4 5-8" />
                  <rect x="8" y="9" width="2" height="6" />
                  <rect x="12" y="11" width="2" height="4" />
                  <rect x="16" y="7" width="2" height="8" />
                </svg>
              </div>
              <h2 className="text-xl font-bold text-purple-800 mb-4">Advanced Technical Analysis</h2>
              <ul className="space-y-3">
                <li className="flex items-start">
                  <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-purple-600 mr-2 mt-1">
                    <polyline points="20 6 9 17 4 12" />
                  </svg>
                  <span className="text-purple-700">Detect key candlestick patterns</span>
                </li>
                <li className="flex items-start">
                  <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-purple-600 mr-2 mt-1">
                    <polyline points="20 6 9 17 4 12" />
                  </svg>
                  <span className="text-purple-700">Analyze support and resistance levels</span>
                </li>
                <li className="flex items-start">
                  <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-purple-600 mr-2 mt-1">
                    <polyline points="20 6 9 17 4 12" />
                  </svg>
                  <span className="text-purple-700">Get precise buy/sell signals</span>
                </li>
                <li className="flex items-start">
                  <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-purple-600 mr-2 mt-1">
                    <polyline points="20 6 9 17 4 12" />
                  </svg>
                  <span className="text-purple-700">Track your prediction history</span>
                </li>
              </ul>
            </motion.div>
          </div>
        </div>
        
        {/* Features Section */}
        <div className="mt-24">
          <h2 className="text-3xl font-bold text-purple-900 text-center mb-12">How It Works</h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-10">
            {[
              {
                title: "Upload Chart",
                description: "Simply upload your candlestick chart from any trading platform or exchange.",
                icon: (
                  <svg xmlns="http://www.w3.org/2000/svg" width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                    <polyline points="17 8 12 3 7 8" />
                    <line x1="12" y1="3" x2="12" y2="15" />
                  </svg>
                )
              },
              {
                title: "Select Timeframe",
                description: "Choose the appropriate timeframe for your trading strategy.",
                icon: (
                  <svg xmlns="http://www.w3.org/2000/svg" width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                    <circle cx="12" cy="12" r="10" />
                    <polyline points="12 6 12 12 16 14" />
                  </svg>
                )
              },
              {
                title: "Get Predictions",
                description: "Receive AI-powered analysis with buy, sell, or hold recommendations.",
                icon: (
                  <svg xmlns="http://www.w3.org/2000/svg" width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5L14.5 2z" />
                    <polyline points="14 2 14 8 20 8" />
                    <path d="m9 15 2 2 4-4" />
                  </svg>
                )
              }
            ].map((feature, index) => (
              <motion.div 
                key={feature.title}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: index * 0.1 }}
                className="bg-white p-6 rounded-lg shadow-md border border-purple-200 flex flex-col items-center text-center"
              >
                <div className="bg-purple-100 rounded-full w-20 h-20 flex items-center justify-center text-purple-800 mb-4">
                  {feature.icon}
                </div>
                <h3 className="text-xl font-bold text-purple-800 mb-2">{feature.title}</h3>
                <p className="text-purple-700">{feature.description}</p>
              </motion.div>
            ))}
          </div>
        </div>
        
        {/* Call to Action */}
        <div className="mt-24 text-center">
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.5 }}
            className="bg-gradient-to-r from-purple-900 to-purple-700 p-10 rounded-xl shadow-xl"
          >
            <h2 className="text-3xl font-bold text-white mb-4">Ready to improve your trading decisions?</h2>
            <p className="text-purple-200 mb-8 max-w-lg mx-auto">
              Get started now with our AI-powered chart analysis platform. No registration required.
            </p>
            <Link href="/dashboard">
              <Button className="bg-white text-purple-900 hover:bg-purple-100 px-8 py-6 text-lg">
                Start Analyzing Now
              </Button>
            </Link>
          </motion.div>
        </div>
      </div>
      
      {/* Footer */}
      <footer className="bg-purple-900 py-8 mt-24">
        <div className="container mx-auto px-4">
          <div className="flex flex-col md:flex-row justify-between items-center">
            <div className="mb-6 md:mb-0">
              <div className="flex items-center space-x-2">
                <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-purple-300">
                  <path d="M12 2v20M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6" />
                </svg>
                <span className="font-bold text-xl text-white">CryptoPredict</span>
              </div>
              <p className="text-purple-300 mt-2">Advanced chart analysis for smarter trading</p>
            </div>
            <div className="grid grid-cols-2 md:grid-cols-3 gap-8">
              <div>
                <h3 className="text-white font-medium mb-3">Features</h3>
                <ul className="space-y-2">
                  <li><Link href="/dashboard" className="text-purple-300 hover:text-white">Chart Upload</Link></li>
                  <li><Link href="/results" className="text-purple-300 hover:text-white">Analysis</Link></li>
                  <li><Link href="/history" className="text-purple-300 hover:text-white">History</Link></li>
                </ul>
              </div>
              <div>
                <h3 className="text-white font-medium mb-3">Resources</h3>
                <ul className="space-y-2">
                  <li><Link href="/education" className="text-purple-300 hover:text-white">Education</Link></li>
                  <li><Link href="/education" className="text-purple-300 hover:text-white">Chart Patterns</Link></li>
                  <li><Link href="/education" className="text-purple-300 hover:text-white">Strategies</Link></li>
                </ul>
              </div>
              <div className="col-span-2 md:col-span-1">
                <h3 className="text-white font-medium mb-3">About</h3>
                <p className="text-purple-300">
                  CryptoPredict provides AI-powered technical analysis for cryptocurrency traders.
                </p>
              </div>
            </div>
          </div>
          <div className="border-t border-purple-800 mt-8 pt-8 flex flex-col md:flex-row justify-between items-center">
            <p className="text-purple-300">© {new Date().getFullYear()} CryptoPredict. All rights reserved.</p>
            <div className="flex space-x-4 mt-4 md:mt-0">
              <a href="#" className="text-purple-300 hover:text-white">
                <span className="sr-only">Twitter</span>
                <svg className="h-6 w-6" fill="currentColor" viewBox="0 0 24 24" aria-hidden="true">
                  <path d="M8.29 20.251c7.547 0 11.675-6.253 11.675-11.675 0-.178 0-.355-.012-.53A8.348 8.348 0 0022 5.92a8.19 8.19 0 01-2.357.646 4.118 4.118 0 001.804-2.27 8.224 8.224 0 01-2.605.996 4.107 4.107 0 00-6.993 3.743 11.65 11.65 0 01-8.457-4.287 4.106 4.106 0 001.27 5.477A4.072 4.072 0 012.8 9.713v.052a4.105 4.105 0 003.292 4.022 4.095 4.095 0 01-1.853.07 4.108 4.108 0 003.834 2.85A8.233 8.233 0 012 18.407a11.616 11.616 0 006.29 1.84" />
                </svg>
              </a>
              <a href="#" className="text-purple-300 hover:text-white">
                <span className="sr-only">GitHub</span>
                <svg className="h-6 w-6" fill="currentColor" viewBox="0 0 24 24" aria-hidden="true">
                  <path fillRule="evenodd" d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.531 1.032 1.531 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0112 6.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0022 12.017C22 6.484 17.522 2 12 2z" clipRule="evenodd" />
                </svg>
              </a>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}
