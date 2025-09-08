import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { useQuery } from '@tanstack/react-query';
import { Card, CardHeader, CardContent, CardTitle } from '../ui/Card';
import { Button } from '../ui/Button';
import { Badge } from '../ui/Badge';
import apiClient from '../../services/api';
import { PestRequest } from '../../types/api';

interface PestDetectionProps {
  farmId: string;
}

export const PestDetection: React.FC<PestDetectionProps> = ({ farmId }) => {
  const [selectedCrop, setSelectedCrop] = useState('wheat');

  const { data: pestData, isLoading, error } = useQuery({
    queryKey: ['pest', farmId, selectedCrop],
    queryFn: () => apiClient.getPestAnalysis({
      farm_id: farmId,
      crop_type: selectedCrop,
      images: [], // In real app, this would come from image upload
      sensor_data: {
        temperature: 25.5,
        humidity: 70,
        soil_moisture: 0.6
      }
    }),
    refetchInterval: 30000, // Refetch every 30 seconds
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
          <p>Failed to load pest detection data</p>
        </div>
      </Card>
    );
  }

  const pestDetection = pestData?.data?.pest_detection;
  const diseaseDetection = pestData?.data?.disease_detection;
  const recommendations = pestData?.data?.recommendations || [];
  const riskLevel = pestData?.data?.risk_level;

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
    >
      <Card className="pest-detection">
        <CardHeader className="flex flex-row items-center justify-between">
          <div>
            <CardTitle>Pest Detection System</CardTitle>
            <p className="text-sm text-gray-500 mt-1">AI-powered pest and disease monitoring</p>
          </div>
          <div className="flex items-center space-x-2">
            <Badge 
              variant={riskLevel === 'high' ? 'error' : riskLevel === 'medium' ? 'warning' : 'success'}
              animated
            >
              {riskLevel?.toUpperCase() || 'LOW'} RISK
            </Badge>
          </div>
        </CardHeader>
        <CardContent>
          <div className="space-y-6">
            {/* Crop Selection */}
            <div className="flex space-x-2">
              {['wheat', 'corn', 'soybean', 'tomato'].map((crop) => (
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

            {/* Detection Results */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* Pest Detection */}
              <div className="space-y-4">
                <h4 className="text-sm font-medium text-gray-900">Pest Detection</h4>
                {pestDetection?.detected ? (
                  <div className="p-4 bg-red-50 border border-red-200 rounded-lg">
                    <div className="flex items-center space-x-2 mb-2">
                      <div className="h-3 w-3 bg-red-500 rounded-full"></div>
                      <span className="text-sm font-medium text-red-800">Pest Detected</span>
                    </div>
                    <p className="text-sm text-red-700">
                      <strong>Type:</strong> {pestDetection.pest_type}
                    </p>
                    <p className="text-sm text-red-700">
                      <strong>Severity:</strong> {pestDetection.severity}
                    </p>
                    <p className="text-sm text-red-700">
                      <strong>Confidence:</strong> {Math.round(pestDetection.confidence * 100)}%
                    </p>
                  </div>
                ) : (
                  <div className="p-4 bg-green-50 border border-green-200 rounded-lg">
                    <div className="flex items-center space-x-2">
                      <div className="h-3 w-3 bg-green-500 rounded-full"></div>
                      <span className="text-sm font-medium text-green-800">No Pests Detected</span>
                    </div>
                    <p className="text-sm text-green-700 mt-1">
                      Your crops are healthy and pest-free
                    </p>
                  </div>
                )}
              </div>

              {/* Disease Detection */}
              <div className="space-y-4">
                <h4 className="text-sm font-medium text-gray-900">Disease Detection</h4>
                {diseaseDetection?.detected ? (
                  <div className="p-4 bg-yellow-50 border border-yellow-200 rounded-lg">
                    <div className="flex items-center space-x-2 mb-2">
                      <div className="h-3 w-3 bg-yellow-500 rounded-full"></div>
                      <span className="text-sm font-medium text-yellow-800">Disease Detected</span>
                    </div>
                    <p className="text-sm text-yellow-700">
                      <strong>Type:</strong> {diseaseDetection.disease_type}
                    </p>
                    <p className="text-sm text-yellow-700">
                      <strong>Confidence:</strong> {Math.round(diseaseDetection.confidence * 100)}%
                    </p>
                  </div>
                ) : (
                  <div className="p-4 bg-green-50 border border-green-200 rounded-lg">
                    <div className="flex items-center space-x-2">
                      <div className="h-3 w-3 bg-green-500 rounded-full"></div>
                      <span className="text-sm font-medium text-green-800">No Diseases Detected</span>
                    </div>
                    <p className="text-sm text-green-700 mt-1">
                      No signs of plant diseases
                    </p>
                  </div>
                )}
              </div>
            </div>

            {/* Sensor Data */}
            <div className="p-4 bg-gray-50 rounded-lg">
              <h4 className="text-sm font-medium text-gray-900 mb-3">Environmental Conditions</h4>
              <div className="grid grid-cols-3 gap-4">
                <div className="text-center">
                  <p className="text-2xl font-bold text-blue-600">25.5°C</p>
                  <p className="text-xs text-gray-600">Temperature</p>
                </div>
                <div className="text-center">
                  <p className="text-2xl font-bold text-cyan-600">70%</p>
                  <p className="text-xs text-gray-600">Humidity</p>
                </div>
                <div className="text-center">
                  <p className="text-2xl font-bold text-green-600">60%</p>
                  <p className="text-xs text-gray-600">Soil Moisture</p>
                </div>
              </div>
            </div>

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
                Upload Images
              </Button>
              <Button variant="outline" size="sm">
                View History
              </Button>
              <Button variant="outline" size="sm">
                Export Report
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>
    </motion.div>
  );
};
