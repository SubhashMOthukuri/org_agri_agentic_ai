import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { useQuery } from '@tanstack/react-query';
import { Card, CardHeader, CardContent, CardTitle } from '../ui/Card';
import { Button } from '../ui/Button';
import { PriceChart } from '../charts/PriceChart';
import apiClient from '../../services/api';
import { MarketRequest } from '../../types/api';

interface MarketIntelligenceProps {
  farmId: string;
}

export const MarketIntelligence: React.FC<MarketIntelligenceProps> = ({ farmId }) => {
  const [selectedCrop, setSelectedCrop] = useState<'wheat' | 'corn' | 'soybean' | 'tomato'>('wheat');
  const [timeframe, setTimeframe] = useState<'1m' | '3m' | '6m' | '1y'>('6m');

  const { data: marketData, isLoading, error } = useQuery({
    queryKey: ['market', selectedCrop, timeframe],
    queryFn: () => apiClient.getMarketAnalysis({
      crop_type: selectedCrop,
      region: 'North America',
      analysis_type: 'price_prediction',
      timeframe: timeframe
    }),
    refetchInterval: 60000, // Refetch every minute
  });

  if (isLoading) {
    return (
      <Card className="p-6">
        <div className="animate-pulse">
          <div className="h-4 bg-gray-200 rounded w-1/4 mb-4"></div>
          <div className="h-32 bg-gray-200 rounded"></div>
        </div>
      </Card>
    );
  }

  if (error) {
    return (
      <Card className="p-6">
        <div className="text-red-600 text-center">
          <p>Failed to load market data</p>
        </div>
      </Card>
    );
  }

  const currentPrice = marketData?.data?.current_price;
  const pricePrediction = marketData?.data?.price_prediction;
  const marketTrends = marketData?.data?.market_trends;
  const recommendations = marketData?.data?.recommendations || [];

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
    >
      <Card className="market-intelligence">
        <CardHeader className="flex flex-row items-center justify-between">
          <div>
            <CardTitle>Market Intelligence</CardTitle>
            <p className="text-sm text-gray-500 mt-1">Real-time price analysis and predictions</p>
          </div>
          <div className="flex space-x-2">
            <Button
              variant={timeframe === '1m' ? 'primary' : 'outline'}
              size="sm"
              onClick={() => setTimeframe('1m')}
            >
              1M
            </Button>
            <Button
              variant={timeframe === '3m' ? 'primary' : 'outline'}
              size="sm"
              onClick={() => setTimeframe('3m')}
            >
              3M
            </Button>
            <Button
              variant={timeframe === '6m' ? 'primary' : 'outline'}
              size="sm"
              onClick={() => setTimeframe('6m')}
            >
              6M
            </Button>
            <Button
              variant={timeframe === '1y' ? 'primary' : 'outline'}
              size="sm"
              onClick={() => setTimeframe('1y')}
            >
              1Y
            </Button>
          </div>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="space-y-4">
              <div className="flex space-x-2">
                {(['wheat', 'corn', 'soybean', 'tomato'] as const).map((crop) => (
                  <Button
                    key={crop}
                    variant={selectedCrop === crop ? 'primary' : 'outline'}
                    size="sm"
                    onClick={() => setSelectedCrop(crop)}
                  >
                    {crop.charAt(0).toUpperCase() + crop.slice(1)}
                  </Button>
                ))}
              </div>
              
              {currentPrice && (
                <div className="space-y-4">
                  <div className="p-4 bg-gray-50 rounded-lg">
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-gray-600">Current Price</span>
                      <span className="text-2xl font-bold text-gray-900">
                        ${currentPrice}/ton
                      </span>
                    </div>
                    {marketTrends && (
                      <div className={`flex items-center mt-1 text-sm ${
                        marketTrends.trend === 'increasing' ? 'text-green-600' : 
                        marketTrends.trend === 'decreasing' ? 'text-red-600' : 'text-gray-600'
                      }`}>
                        {marketTrends.trend === 'increasing' ? '↗️' : 
                         marketTrends.trend === 'decreasing' ? '↘️' : '→'} 
                        {marketTrends.trend} ({Math.round(marketTrends.confidence * 100)}% confidence)
                      </div>
                    )}
                  </div>
                  
                  {pricePrediction && (
                    <div className="grid grid-cols-3 gap-2">
                      <PredictionCard
                        period="1 Month"
                        price={pricePrediction['1_month']}
                        currentPrice={currentPrice}
                      />
                      <PredictionCard
                        period="3 Months"
                        price={pricePrediction['3_months']}
                        currentPrice={currentPrice}
                      />
                      <PredictionCard
                        period="6 Months"
                        price={pricePrediction['6_months']}
                        currentPrice={currentPrice}
                      />
                    </div>
                  )}
                </div>
              )}

              {recommendations.length > 0 && (
                <div className="p-4 bg-blue-50 rounded-lg">
                  <h4 className="text-sm font-medium text-blue-900 mb-2">AI Recommendations</h4>
                  <ul className="space-y-1">
                    {recommendations.map((rec, index) => (
                      <li key={index} className="text-sm text-blue-800">
                        • {rec}
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
            
            <div className="h-64">
              <PriceChart 
                crop={selectedCrop} 
                timeframe={timeframe}
                data={marketData?.data}
              />
            </div>
          </div>
        </CardContent>
      </Card>
    </motion.div>
  );
};

const PredictionCard: React.FC<{
  period: string;
  price: number;
  currentPrice: number;
}> = ({ period, price, currentPrice }) => {
  const change = ((price - currentPrice) / currentPrice) * 100;
  
  return (
    <div className="p-3 bg-white border border-gray-200 rounded-lg">
      <p className="text-xs text-gray-600">{period}</p>
      <p className="text-lg font-bold text-gray-900">${price.toFixed(2)}</p>
      <p className={`text-xs ${change > 0 ? 'text-green-600' : 'text-red-600'}`}>
        {change > 0 ? '+' : ''}{change.toFixed(1)}%
      </p>
    </div>
  );
};
