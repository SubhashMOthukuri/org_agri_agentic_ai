import axios, { AxiosInstance, AxiosResponse, AxiosError } from 'axios';
import { config } from '../config/environment';
import {
  AuthResponse,
  LoginRequest,
  User,
  WeatherRequest,
  WeatherResponse,
  MarketRequest,
  MarketResponse,
  PestRequest,
  PestResponse,
  PredictionRequest,
  PredictionResponse,
  DecisionRequest,
  DecisionResponse,
  FarmsResponse,
  MetricsResponse,
  HealthResponse,
  ApiError
} from '../types/api';

class ApiClient {
  private client: AxiosInstance;

  constructor() {
    this.client = axios.create({
      baseURL: config.apiBaseUrl,
      timeout: 30000,
      headers: {
        'Content-Type': 'application/json',
      },
    });

    this.setupInterceptors();
  }

  private setupInterceptors() {
    // Request interceptor to add auth token
    this.client.interceptors.request.use(
      (config) => {
        const token = localStorage.getItem('token');
        if (token) {
          config.headers.Authorization = `Bearer ${token}`;
        }
        return config;
      },
      (error) => Promise.reject(error)
    );

    // Response interceptor for error handling
    this.client.interceptors.response.use(
      (response: AxiosResponse) => response,
      (error: AxiosError) => {
        if (error.response?.status === 401) {
          // Token expired or invalid
          localStorage.removeItem('token');
          localStorage.removeItem('user');
          window.location.href = '/login';
        }
        return Promise.reject(this.handleError(error));
      }
    );
  }

  private handleError(error: AxiosError): ApiError {
    const apiError: ApiError = {
      error: 'UNKNOWN_ERROR',
      message: 'An unexpected error occurred',
      status_code: 500,
    };

    if (error.response) {
      // Server responded with error status
      apiError.status_code = error.response.status;
      apiError.message = error.response.data?.message || error.message;
      apiError.error = error.response.data?.error || 'API_ERROR';
      apiError.details = error.response.data?.details;
    } else if (error.request) {
      // Request was made but no response received
      apiError.message = 'Network error - please check your connection';
      apiError.error = 'NETWORK_ERROR';
    }

    return apiError;
  }

  // Authentication methods
  async login(credentials: LoginRequest): Promise<AuthResponse> {
    const response = await this.client.post<AuthResponse>('/auth/login', credentials);
    return response.data;
  }

  async getCurrentUser(): Promise<User> {
    const response = await this.client.get<User>('/auth/me');
    return response.data;
  }

  // AI Agent methods
  async getWeatherAnalysis(request: WeatherRequest): Promise<WeatherResponse> {
    const response = await this.client.post<WeatherResponse>('/agents/weather', request);
    return response.data;
  }

  async getMarketAnalysis(request: MarketRequest): Promise<MarketResponse> {
    const response = await this.client.post<MarketResponse>('/agents/market', request);
    return response.data;
  }

  async getPestAnalysis(request: PestRequest): Promise<PestResponse> {
    const response = await this.client.post<PestResponse>('/agents/pest', request);
    return response.data;
  }

  async getYieldPrediction(request: PredictionRequest): Promise<PredictionResponse> {
    const response = await this.client.post<PredictionResponse>('/agents/prediction', request);
    return response.data;
  }

  async getDecisionSupport(request: DecisionRequest): Promise<DecisionResponse> {
    const response = await this.client.post<DecisionResponse>('/agents/decision', request);
    return response.data;
  }

  async getComprehensiveAnalysis(farmId: string, scope: 'comprehensive' | 'quick' | 'detailed' = 'comprehensive') {
    const response = await this.client.post('/agents/analyze', {
      farm_id: farmId,
      analysis_scope: scope
    });
    return response.data;
  }

  // Data access methods
  async getFarms(page: number = 1, limit: number = 10, search?: string, status?: string): Promise<FarmsResponse> {
    const params = new URLSearchParams({
      page: page.toString(),
      limit: limit.toString(),
    });
    
    if (search) params.append('search', search);
    if (status) params.append('status', status);

    const response = await this.client.get<FarmsResponse>(`/data/farms?${params}`);
    return response.data;
  }

  async getFarm(farmId: string) {
    const response = await this.client.get(`/data/farms/${farmId}`);
    return response.data;
  }

  async getCrops() {
    const response = await this.client.get('/data/crops');
    return response.data;
  }

  // Metrics methods
  async getMetrics(): Promise<MetricsResponse> {
    const response = await this.client.get<MetricsResponse>('/metrics');
    return response.data;
  }

  async getCriticalMetrics() {
    const response = await this.client.get('/metrics/critical');
    return response.data;
  }

  // Health check
  async getHealth(): Promise<HealthResponse> {
    const response = await this.client.get<HealthResponse>('/health');
    return response.data;
  }

  // Utility methods
  setAuthToken(token: string) {
    localStorage.setItem('token', token);
  }

  removeAuthToken() {
    localStorage.removeItem('token');
    localStorage.removeItem('user');
  }

  isAuthenticated(): boolean {
    return !!localStorage.getItem('token');
  }
}

// Create and export a singleton instance
export const apiClient = new ApiClient();
export default apiClient;
