// API Types based on backend documentation

export interface User {
  user_id: string;
  username: string;
  email: string;
  role: string;
  permissions: string[];
}

export interface AuthResponse {
  access_token: string;
  token_type: string;
  expires_in: number;
}

export interface LoginRequest {
  username: string;
  password: string;
}

// Weather API Types
export interface WeatherLocation {
  latitude: number;
  longitude: number;
}

export interface WeatherRequest {
  farm_id: string;
  location?: WeatherLocation;
  analysis_type: 'forecast' | 'current' | 'historical';
  timeframe?: '7d' | '30d' | '90d';
}

export interface CurrentWeather {
  temperature: number;
  humidity: number;
  precipitation: number;
  wind_speed: number;
  uv_index?: number;
  timestamp: string;
}

export interface WeatherForecast {
  date: string;
  temperature: number;
  precipitation_probability: number;
  recommendations: string[];
}

export interface WeatherRiskAssessment {
  drought_risk: 'low' | 'medium' | 'high';
  flood_risk: 'low' | 'medium' | 'high';
  temperature_stress: 'none' | 'low' | 'medium' | 'high';
}

export interface WeatherResponse {
  status: string;
  data: {
    current_weather: CurrentWeather;
    forecast: WeatherForecast[];
    risk_assessment: WeatherRiskAssessment;
  };
}

// Market API Types
export interface MarketRequest {
  crop_type: 'wheat' | 'corn' | 'soybean' | 'tomato';
  region: string;
  analysis_type: 'price_prediction' | 'market_trends' | 'supply_demand';
  timeframe: '1m' | '3m' | '6m' | '1y';
}

export interface PricePrediction {
  '1_month': number;
  '3_months': number;
  '6_months': number;
}

export interface MarketTrends {
  trend: 'increasing' | 'decreasing' | 'stable';
  volatility: 'low' | 'medium' | 'high';
  confidence: number;
}

export interface MarketResponse {
  status: string;
  data: {
    current_price: number;
    price_prediction: PricePrediction;
    market_trends: MarketTrends;
    recommendations: string[];
  };
}

// Pest Detection API Types
export interface SensorData {
  temperature: number;
  humidity: number;
  soil_moisture: number;
}

export interface PestRequest {
  farm_id: string;
  crop_type: string;
  images: string[];
  sensor_data: SensorData;
}

export interface PestDetection {
  detected: boolean;
  pest_type: string | null;
  severity: 'low' | 'medium' | 'high';
  confidence: number;
}

export interface DiseaseDetection {
  detected: boolean;
  disease_type: string | null;
  confidence: number;
}

export interface PestResponse {
  status: string;
  data: {
    pest_detection: PestDetection;
    disease_detection: DiseaseDetection;
    recommendations: string[];
    risk_level: 'low' | 'medium' | 'high';
  };
}

// Yield Prediction API Types
export interface HistoricalData {
  previous_yields: number[];
  weather_conditions: string[];
}

export interface PredictionRequest {
  farm_id: string;
  crop_type: string;
  planting_date: string;
  expected_harvest: string;
  historical_data: HistoricalData;
}

export interface YieldFactors {
  weather_impact: number;
  soil_quality: number;
  pest_pressure: number;
}

export interface YieldPrediction {
  predicted_yield: number;
  confidence_interval: [number, number];
  factors: YieldFactors;
}

export interface SupplyChainRisk {
  overall_risk: 'low' | 'medium' | 'high';
  logistics_risk: 'low' | 'medium' | 'high';
  market_risk: 'low' | 'medium' | 'high';
}

export interface PredictionResponse {
  status: string;
  data: {
    yield_prediction: YieldPrediction;
    supply_chain_risk: SupplyChainRisk;
    recommendations: string[];
  };
}

// Decision API Types
export interface DecisionContext {
  current_conditions: string;
  goals: string[];
  constraints: string[];
}

export interface DecisionRequest {
  farm_id: string;
  decision_type: 'planting' | 'harvesting' | 'irrigation' | 'pest_control';
  context: DecisionContext;
}

export interface DecisionAlternative {
  action: string;
  pros: string[];
  cons: string[];
}

export interface DecisionResponse {
  status: string;
  data: {
    decision: {
      recommended_action: string;
      confidence: number;
      reasoning: string;
      expected_outcome: string;
    };
    alternatives: DecisionAlternative[];
    implementation_plan: string[];
  };
}

// Farm Data Types
export interface FarmLocation {
  latitude: number;
  longitude: number;
  address: string;
}

export interface Farm {
  farm_id: string;
  name: string;
  location: FarmLocation;
  crops: string[];
  status: 'active' | 'inactive' | 'maintenance';
  last_updated: string;
}

export interface FarmsResponse {
  status: string;
  data: {
    farms: Farm[];
    pagination: {
      page: number;
      limit: number;
      total: number;
      pages: number;
    };
  };
}

// Metrics Types
export interface DataQuality {
  completeness: number;
  consistency: number;
  accuracy: number;
  status: 'EXCELLENT' | 'GOOD' | 'WARNING' | 'CRITICAL';
}

export interface PerformanceMetrics {
  records_per_second: number;
  memory_usage: number;
  cpu_usage: number;
  status: 'EXCELLENT' | 'GOOD' | 'WARNING' | 'CRITICAL';
}

export interface BusinessMetrics {
  enterprise_readiness: number;
  total_records: number;
  active_farms: number;
  status: 'EXCELLENT' | 'GOOD' | 'WARNING' | 'CRITICAL';
}

export interface MetricsResponse {
  status: string;
  data: {
    data_quality: DataQuality;
    performance: PerformanceMetrics;
    business: BusinessMetrics;
  };
}

// Health Check Types
export interface HealthServices {
  database: 'healthy' | 'unhealthy';
  redis: 'healthy' | 'unhealthy';
  agents: 'healthy' | 'unhealthy';
}

export interface HealthResponse {
  status: 'healthy' | 'unhealthy';
  timestamp: string;
  version: string;
  uptime: number;
  services: HealthServices;
}

// WebSocket Types
export interface WebSocketMessage {
  type: 'weather_update' | 'pest_alert' | 'market_update' | 'system_alert';
  data: any;
  timestamp: string;
}

// Error Types
export interface ApiError {
  error: string;
  message: string;
  details?: any;
  status_code: number;
}
