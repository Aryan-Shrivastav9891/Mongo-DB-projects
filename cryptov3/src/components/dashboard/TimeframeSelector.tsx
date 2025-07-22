"use client";

import { useState } from 'react';
import { motion } from 'framer-motion';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { TimeframeOption, timeframeOptions } from '@/lib/theme';

interface TimeframeSelectorProps {
  selectedTimeframe: TimeframeOption;
  onTimeframeChange: (timeframe: TimeframeOption) => void;
}

export function TimeframeSelector({
  selectedTimeframe,
  onTimeframeChange,
}: TimeframeSelectorProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay: 0.1 }}
    >
      <Card className="w-full border-purple-200 bg-gradient-to-br from-white to-purple-50 shadow-md">
        <CardHeader className="border-b border-purple-100">
          <CardTitle className="text-center text-purple-800">Select Timeframe</CardTitle>
        </CardHeader>
        <CardContent>
          <Tabs defaultValue={selectedTimeframe} onValueChange={(value) => onTimeframeChange(value as TimeframeOption)}>
            <TabsList className="w-full grid grid-cols-5 bg-purple-100">
              {timeframeOptions.map((option) => (
                <TabsTrigger key={option} value={option} className="data-[state=active]:bg-purple-700 data-[state=active]:text-white">
                  {option}
                </TabsTrigger>
              ))}
            </TabsList>
            {timeframeOptions.map((option) => (
              <TabsContent key={option} value={option} className="p-2 text-center">
                <p className="text-sm text-purple-700 mt-2">
                  {getTimeframeDescription(option)}
                </p>
              </TabsContent>
            ))}
          </Tabs>
        </CardContent>
      </Card>
    </motion.div>
  );
}

function getTimeframeDescription(timeframe: TimeframeOption): string {
  switch (timeframe) {
    case '15m':
      return 'Short-term analysis for intraday traders';
    case '1h':
      return 'Balanced timeframe for day traders';
    case '4h':
      return 'Medium-term analysis for swing traders';
    case '1d':
      return 'Daily analysis for position traders';
    case '7d':
      return 'Long-term trend analysis for investors';
    default:
      return '';
  }
}
