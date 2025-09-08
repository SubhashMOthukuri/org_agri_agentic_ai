import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { useQuery } from '@tanstack/react-query';
import { Card, CardHeader, CardContent, CardTitle } from '../ui/Card';
import { Button } from '../ui/Button';
import { Badge } from '../ui/Badge';
import apiClient from '../../services/api';
import { PredictionRequest } from '../../types/api';

interface YieldPredictionProps {
  farmId: string;
}

export const YieldPrediction: React.FC<YieldPredictionProps> = ({ farmId }) => {
  const [selectedCrop, setSelectedCrop] = useState('wheat');
  const [plantingDate, setPlantingDate] = useState('2024-03-15');
  const [harvestDate, setHarvestDate] = useState('2024-08-15');

  const { data: predictionData, isLoading, error } = useQuery({
    queryKey: ['yield-prediction', farmId, selectedCrop, plantingDate, harvestDate],
    queryFn: () => apiClient.getYieldPrediction({
      farm_id: farmId,
      crop_type: selectedCrop,
      planting_date: plantingDate,
      expected_harvest: harvestDate,
      historical_data: {
        previous_yields: [1200, 1350, 1280],
        weather_conditions: ['normal', 'drought', 'normal']
      }
    }),
    refetchInterval: 300000, // Refetch every 5 minutes
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
          <p>Failed to load yield prediction data</p>
        </div>
      </Card>
    );
  }

  const yieldPrediction = predictionData?.data?.yield_prediction;
  const supplyChainRisk = predictionData?.data?.supply_chain_risk;
  const recommendations = predictionData?.data?.recommendations || [];

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
    >
      <Card className="yield-prediction">
        <CardHeader className="flex flex-row items-center justify-between">
          <div>
            <CardTitle>Yield Prediction</CardTitle>
            <p className="text-sm text-gray-500 mt-1">AI-powered yield forecasting</p>
          </div>
          <Badge variant="info">AI Prediction</Badge>
        </CardHeader>
        <CardContent>
          <div className="space-y-6">
            {/* Crop and Date Selection */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Crop Type</label>
                <select
                  value={selectedCrop}
                  onChange={(e) => setSelectedCrop(e.target.value)}
                  className="w-full rounded-lg border-gray-300 shadow-sm focus:border-green-500 focus:ring-green-500"
                >
                  <option value="wheat">Wheat</option>
                  <option value="corn">Corn</option>
                  <option value="soybean">Soybean</option>
                  <option value="tomato">Tomato</option>
                </select>
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Planting Date</label>
                <input
                  type="date"
                  value={plantingDate}
                  onChange={(e) => setPlantingDate(e.target.value)}
                  className="w-full rounded-lg border-gray-300 shadow-sm focus:border-green-500 focus:ring-green-500"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Expected Harvest</label>
                <input
                  type="date"
                  value={harvestDate}
                  onChange={(e) => setHarvestDate(e.target.value)}
                  className="w-full rounded-lg border-gray-300 shadow-sm focus:border-green-500 focus:ring-green-500"
                />
              </div>
            </div>

            {/* Yield Prediction Results */}
            {yieldPrediction && (
              <div className="space-y-4">
                <div className="p-6 bg-gradient-to-r from-green-50 to-blue-50 rounded-lg border border-green-200">
                  <div className="text-center">
                    <h3 className="text-2xl font-bold text-gray-900 mb-2">
                      {yieldPrediction.predicted_yield.toFixed(1)} kg/acre
                    </h3>
                    <p className="text-sm text-gray-600">Predicted Yield</p>
                    <div className="mt-2 text-xs text-gray-500">
                      Confidence: {yieldPrediction.confidence_interval[0].toFixed(0)} - {yieldPrediction.confidence_interval[1].toFixed(0)} kg/acre
                    </div>
                  </div>
                </div>

                {/* Yield Factors */}
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <FactorCard
                    label="Weather Impact"
                    value={yieldPrediction.factors.weather_impact}
                    color="blue"
                  />
                  <FactorCard
                    label="Soil Quality"
                    value={yieldPrediction.factors.soil_quality}
                    color="green"
                  />
                  <FactorCard
                    label="Pest Pressure"
                    value={yieldPrediction.factors.pest_pressure}
                    color="red"
                  />
                </div>
              </div>
            )}

            {/* Supply Chain Risk */}
            {supplyChainRisk && (
              <div className="p-4 bg-gray-50 rounded-lg">
                <h4 className="text-sm font-medium text-gray-900 mb-3">Supply Chain Risk Assessment</h4>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <RiskIndicator
                    label="Overall Risk"
                    level={supplyChainRisk.overall_risk}
                  />
                  <RiskIndicator
                    label="Logistics Risk"
                    level={supplyChainRisk.logistics_risk}
                  />
                  <RiskIndicator
                    label="Market Risk"
                    level={supplyChainRisk.market_risk}
                  />
                </div>
              </div>
            )}

            {/* Recommendations */}
            {recommendations.length > 0 && (
              <div className="p-4 bg-blue-50 rounded-lg">
                <h4 className="text-sm font-medium text-blue-900 mb-3">AI Recommendations</h4>
                <ul className="space-y-2">
                  {recommendations.map((rec, index) => (
                    <motion.li
                      key={index}
                      initial={{ opacity: 0, x: -10 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: index * 0.1 }}
                      className="flex items-start space-x-2"
                    >
                      <span className="text-blue-600 mt-1">•</span>
                      <span className="text-sm text-blue-800">{rec}</span>
                    </motion.li>
                  ))}
                </ul>
              </div>
            )}

            {/* Action Buttons */}
            <div className="flex space-x-3">
              <Button variant="primary" size="sm">
                Generate Report
              </Button>
              <Button variant="outline" size="sm">
                Compare Scenarios
              </Button>
              <Button variant="outline" size="sm">
                Export Data
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>
    </motion.div>
  );
};

const FactorCard: React.FC<{
  label: string;
  value: number;
  color: string;
}> = ({ label, value, color }) => {
  const percentage = Math.round(value * 100);
  const isPositive = value > 0;
  
  return (
    <div className="p-4 bg-white border border-gray-200 rounded-lg">
      <p className="text-sm text-gray-600 mb-2">{label}</p>
      <div className="flex items-center space-x-2">
        <div className={`text-2xl font-bold ${
          isPositive ? 'text-green-600' : 'text-red-600'
        }`}>
          {isPositive ? '+' : ''}{percentage}%
        </div>
        <div className="flex-1">
          <div className="w-full bg-gray-200 rounded-full h-2">
            <div
              className={`h-2 rounded-full ${
                isPositive ? 'bg-green-500' : 'bg-red-500'
              }`}
              style={{ width: `${Math.abs(percentage)}%` }}
            />
          </div>
        </div>
      </div>
    </div>
  );
};

const RiskIndicator: React.FC<{
  label: string;
  level: string;
}> = ({ label, level }) => {
  const getLevelColor = (level: string) => {
    switch (level) {
      case 'low': return 'green';
      case 'medium': return 'yellow';
      case 'high': return 'red';
      default: return 'gray';
    }
  };

  return (
    <div className="text-center">
      <p className="text-sm text-gray-600 mb-1">{label}</p>
      <div className={`inline-flex items-center px-3 py-1 rounded-full text-xs font-medium bg-${getLevelColor(level)}-100 text-${getLevelColor(level)}-800`}>
        <div className={`h-2 w-2 rounded-full bg-${getLevelColor(level)}-500 mr-2`} />
        {level.toUpperCase()}
      </div>
    </div>
  );
};
