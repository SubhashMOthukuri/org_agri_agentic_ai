import React from 'react';
import { motion } from 'framer-motion';
import { useQuery } from '@tanstack/react-query';
import { Card, CardHeader, CardContent, CardTitle } from '../ui/Card';
import { Badge } from '../ui/Badge';
import { WeatherChart } from '../charts/WeatherChart';
import apiClient from '../../services/api';
import { WeatherRequest } from '../../types/api';

interface WeatherDashboardProps {
  farmId: string;
}

export const WeatherDashboard: React.FC<WeatherDashboardProps> = ({ farmId }) => {
  const { data: weatherData, isLoading, error } = useQuery({
    queryKey: ['weather', farmId],
    queryFn: () => apiClient.getWeatherAnalysis({
      farm_id: farmId,
      analysis_type: 'current',
      timeframe: '7d'
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
          <p>Failed to load weather data</p>
        </div>
      </Card>
    );
  }

  const currentWeather = weatherData?.data?.current_weather;
  const forecast = weatherData?.data?.forecast || [];
  const riskAssessment = weatherData?.data?.risk_assessment;

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
    >
      <Card className="weather-dashboard">
        <CardHeader className="flex flex-row items-center justify-between">
          <div>
            <CardTitle>Weather Intelligence</CardTitle>
            <p className="text-sm text-gray-500 mt-1">Real-time weather monitoring</p>
          </div>
          <Badge variant="success" className="animate-pulse">Live</Badge>
        </CardHeader>
        <CardContent>
          {currentWeather && (
            <div className="grid grid-cols-2 md:grid-cols-5 gap-4 mb-6">
              <WeatherMetric
                label="Temperature"
                value={`${currentWeather.temperature}°C`}
                icon="🌡️"
                color="blue"
                change={+2.1}
              />
              <WeatherMetric
                label="Humidity"
                value={`${currentWeather.humidity}%`}
                icon="💧"
                color="cyan"
                change={-3.2}
              />
              <WeatherMetric
                label="Precipitation"
                value={`${currentWeather.precipitation}mm`}
                icon="🌧️"
                color="blue"
                change={+0.8}
              />
              <WeatherMetric
                label="Wind Speed"
                value={`${currentWeather.wind_speed} km/h`}
                icon="💨"
                color="gray"
                change={+1.2}
              />
              <WeatherMetric
                label="UV Index"
                value={`${currentWeather.uv_index || 6.2}`}
                icon="☀️"
                color="yellow"
                change={+0.5}
              />
            </div>
          )}

          <div className="space-y-4">
            <WeatherChart data={weatherData?.data} />
            
            {riskAssessment && (
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <RiskIndicator
                  label="Drought Risk"
                  level={riskAssessment.drought_risk}
                  color="yellow"
                />
                <RiskIndicator
                  label="Flood Risk"
                  level={riskAssessment.flood_risk}
                  color="blue"
                />
                <RiskIndicator
                  label="Temperature Stress"
                  level={riskAssessment.temperature_stress}
                  color="red"
                />
              </div>
            )}
          </div>
        </CardContent>
      </Card>
    </motion.div>
  );
};

const WeatherMetric: React.FC<{
  label: string;
  value: string;
  icon: string;
  color: string;
  change: number;
}> = ({ label, value, icon, color, change }) => (
  <motion.div
    whileHover={{ scale: 1.05 }}
    className={`p-4 rounded-lg bg-${color}-50 border border-${color}-200`}
  >
    <div className="flex items-center space-x-2">
      <span className="text-2xl">{icon}</span>
      <div>
        <p className="text-sm text-gray-600">{label}</p>
        <p className={`text-xl font-bold text-${color}-700`}>{value}</p>
        <p className={`text-xs ${change > 0 ? 'text-green-600' : 'text-red-600'}`}>
          {change > 0 ? '↗️' : '↘️'} {Math.abs(change)}%
        </p>
      </div>
    </div>
  </motion.div>
);

const RiskIndicator: React.FC<{
  label: string;
  level: string;
  color: string;
}> = ({ label, level, color }) => {
  const getLevelColor = (level: string) => {
    switch (level) {
      case 'low': return 'green';
      case 'medium': return 'yellow';
      case 'high': return 'red';
      default: return 'gray';
    }
  };

  return (
    <div className="p-3 rounded-lg bg-gray-50">
      <p className="text-sm text-gray-600">{label}</p>
      <div className="flex items-center space-x-2 mt-1">
        <div className={`h-2 w-2 rounded-full bg-${getLevelColor(level)}-500`} />
        <span className={`text-sm font-medium text-${getLevelColor(level)}-700 capitalize`}>
          {level}
        </span>
      </div>
    </div>
  );
};
