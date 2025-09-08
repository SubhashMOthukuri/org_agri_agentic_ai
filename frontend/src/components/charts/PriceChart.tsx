import React from 'react';
import { motion } from 'framer-motion';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

interface PriceChartProps {
  crop: string;
  timeframe: string;
  data: any;
}

export const PriceChart: React.FC<PriceChartProps> = ({ crop, timeframe, data }) => {
  // Use real API data or fallback to empty array
  const chartData = data?.price_history || [];

  const getTimeframeData = () => {
    switch (timeframe) {
      case '1m':
        return chartData.slice(-1);
      case '3m':
        return chartData.slice(-3);
      case '6m':
        return chartData.slice(-6);
      case '1y':
        return chartData;
      default:
        return chartData.slice(-6);
    }
  };

  const displayData = getTimeframeData();

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.5 }}
      className="h-full"
    >
      <div className="mb-4">
        <h4 className="text-sm font-medium text-gray-900">
          {crop.charAt(0).toUpperCase() + crop.slice(1)} Price Trends
        </h4>
        <p className="text-xs text-gray-500">
          {timeframe === '1m' ? '1 Month' : 
           timeframe === '3m' ? '3 Months' : 
           timeframe === '6m' ? '6 Months' : '1 Year'} view
        </p>
      </div>
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={displayData}>
          <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
          <XAxis 
            dataKey="date" 
            stroke="#666"
            fontSize={12}
            tickFormatter={(value) => {
              const date = new Date(value);
              return date.toLocaleDateString('en-US', { month: 'short' });
            }}
          />
          <YAxis 
            stroke="#666"
            fontSize={12}
            tickFormatter={(value) => `$${value}`}
          />
          <Tooltip 
            formatter={(value: any) => [`$${value}`, 'Price']}
            labelFormatter={(label) => `Date: ${label}`}
            contentStyle={{
              backgroundColor: 'white',
              border: '1px solid #e5e7eb',
              borderRadius: '8px',
              boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)'
            }}
          />
          <Line 
            type="monotone" 
            dataKey="price" 
            stroke="#22c55e" 
            strokeWidth={2}
            dot={{ fill: '#22c55e', strokeWidth: 2, r: 4 }}
            activeDot={{ r: 6, stroke: '#22c55e', strokeWidth: 2 }}
          />
        </LineChart>
      </ResponsiveContainer>
    </motion.div>
  );
};
