"""
Data Models for Organic Agriculture Agentic AI System
Defines MongoDB document schemas and validation rules
"""

from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field, validator
from enum import Enum
import uuid

class SensorType(str, Enum):
    """IoT Sensor Types"""
    TEMPERATURE = "temperature"
    HUMIDITY = "humidity"
    SOIL_MOISTURE = "soil_moisture"
    PH_LEVEL = "ph_level"
    LIGHT_INTENSITY = "light_intensity"
    CO2_LEVEL = "co2_level"
    NITROGEN_LEVEL = "nitrogen_level"
    PHOSPHORUS_LEVEL = "phosphorus_level"
    POTASSIUM_LEVEL = "potassium_level"

class AnomalyType(str, Enum):
    """Anomaly Types"""
    SENSOR_MALFUNCTION = "sensor_malfunction"
    PEST_OUTBREAK = "pest_outbreak"
    DISEASE_DETECTION = "disease_detection"
    WEATHER_EXTREME = "weather_extreme"
    SOIL_CONTAMINATION = "soil_contamination"
    WATER_SHORTAGE = "water_shortage"
    EQUIPMENT_FAILURE = "equipment_failure"
    SUPPLY_CHAIN_DELAY = "supply_chain_delay"

class SeverityLevel(str, Enum):
    """Severity Levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class SupplyChainStatus(str, Enum):
    """Supply Chain Status"""
    PENDING = "pending"
    IN_TRANSIT = "in_transit"
    DELIVERED = "delivered"
    DELAYED = "delayed"
    CANCELLED = "cancelled"

class TransportationMode(str, Enum):
    """Transportation Modes"""
    TRUCK = "truck"
    TRAIN = "train"
    SHIP = "ship"
    AIRPLANE = "airplane"
    DRONE = "drone"

class MarketType(str, Enum):
    """Market Types"""
    LOCAL = "local"
    REGIONAL = "regional"
    NATIONAL = "national"
    INTERNATIONAL = "international"
    ORGANIC_CERTIFIED = "organic_certified"

class CropType(str, Enum):
    """Crop Types"""
    TOMATO = "tomato"
    LETTUCE = "lettuce"
    CARROT = "carrot"
    POTATO = "potato"
    WHEAT = "wheat"
    CORN = "corn"
    SOYBEAN = "soybean"
    RICE = "rice"
    SPINACH = "spinach"
    CUCUMBER = "cucumber"

# Base Models
class Location(BaseModel):
    """Geographic location"""
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    altitude: Optional[float] = None
    address: Optional[str] = None

class CostAnalysis(BaseModel):
    """Cost analysis breakdown"""
    production_cost: float = Field(..., ge=0)
    transportation_cost: float = Field(..., ge=0)
    storage_cost: float = Field(..., ge=0)
    marketing_cost: float = Field(..., ge=0)
    total_cost: float = Field(..., ge=0)
    
    @validator('total_cost', always=True)
    def calculate_total_cost(cls, v, values):
        if 'production_cost' in values and 'transportation_cost' in values:
            return values['production_cost'] + values['transportation_cost'] + values.get('storage_cost', 0) + values.get('marketing_cost', 0)
        return v

class SpectralBands(BaseModel):
    """Satellite spectral bands"""
    red: float = Field(..., ge=0, le=1)
    green: float = Field(..., ge=0, le=1)
    blue: float = Field(..., ge=0, le=1)
    nir: float = Field(..., ge=0, le=1)  # Near Infrared
    swir1: float = Field(..., ge=0, le=1)  # Short Wave Infrared 1
    swir2: float = Field(..., ge=0, le=1)  # Short Wave Infrared 2

# Main Data Models
class IoTSensorData(BaseModel):
    """IoT Sensor Data Model"""
    sensor_id: str = Field(..., min_length=1)
    farm_id: str = Field(..., min_length=1)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    sensor_type: SensorType
    value: float
    unit: str = Field(..., min_length=1)
    location: Optional[Location] = None
    quality_score: Optional[float] = Field(None, ge=0, le=1)
    battery_level: Optional[float] = Field(None, ge=0, le=100)
    signal_strength: Optional[float] = Field(None, ge=0, le=100)
    
    class Config:
        use_enum_values = True

class SatelliteData(BaseModel):
    """Satellite Data Model"""
    farm_id: str = Field(..., min_length=1)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    vegetation_index: float = Field(..., ge=0, le=1)
    spectral_bands: Optional[SpectralBands] = None
    cloud_cover: float = Field(0, ge=0, le=1)
    quality_score: float = Field(1.0, ge=0, le=1)
    satellite_name: Optional[str] = None
    resolution: Optional[float] = Field(None, gt=0)
    
    class Config:
        use_enum_values = True

class SupplyChainData(BaseModel):
    """Supply Chain Data Model"""
    supply_chain_id: str = Field(..., min_length=1)
    farm_id: str = Field(..., min_length=1)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    status: SupplyChainStatus
    transportation_mode: TransportationMode
    estimated_delivery: datetime
    actual_delivery: Optional[datetime] = None
    risk_score: float = Field(0, ge=0, le=1)
    origin_location: Optional[Location] = None
    destination_location: Optional[Location] = None
    cargo_type: str = Field(..., min_length=1)
    cargo_weight: Optional[float] = Field(None, gt=0)
    cargo_volume: Optional[float] = Field(None, gt=0)
    temperature_controlled: bool = False
    special_requirements: Optional[List[str]] = None
    
    class Config:
        use_enum_values = True

class FinancialData(BaseModel):
    """Financial Data Model"""
    farm_id: str = Field(..., min_length=1)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    crop_type: CropType
    market_price: float = Field(..., gt=0)
    market_type: MarketType
    cost_analysis: Optional[CostAnalysis] = None
    profit_margin: Optional[float] = Field(None, ge=-1, le=1)
    demand_forecast: Optional[float] = Field(None, ge=0)
    supply_forecast: Optional[float] = Field(None, ge=0)
    price_volatility: Optional[float] = Field(None, ge=0)
    market_trend: Optional[str] = None  # "increasing", "decreasing", "stable"
    
    class Config:
        use_enum_values = True

class AnomalyData(BaseModel):
    """Anomaly Data Model"""
    anomaly_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    farm_id: str = Field(..., min_length=1)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    anomaly_type: AnomalyType
    severity: SeverityLevel
    description: str = Field(..., min_length=1)
    affected_metrics: List[str] = Field(default_factory=list)
    recommended_actions: List[str] = Field(default_factory=list)
    confidence_score: float = Field(1.0, ge=0, le=1)
    resolved: bool = False
    resolution_timestamp: Optional[datetime] = None
    resolution_notes: Optional[str] = None
    
    class Config:
        use_enum_values = True

# Agent Response Models
class AgentResponse(BaseModel):
    """Base Agent Response Model"""
    agent_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    status: str = Field(..., min_length=1)
    message: str = Field(..., min_length=1)
    data: Optional[Dict[str, Any]] = None
    confidence: float = Field(1.0, ge=0, le=1)
    processing_time_ms: Optional[float] = None

class PredictionResponse(AgentResponse):
    """Prediction Agent Response Model"""
    prediction_type: str = Field(..., min_length=1)
    predicted_value: Union[float, int, str, bool]
    prediction_interval: Optional[Dict[str, float]] = None
    feature_importance: Optional[Dict[str, float]] = None

class RecommendationResponse(AgentResponse):
    """Recommendation Agent Response Model"""
    recommendation_type: str = Field(..., min_length=1)
    priority: str = Field(..., min_length=1)
    expected_impact: str = Field(..., min_length=1)
    implementation_steps: List[str] = Field(default_factory=list)
    required_resources: Optional[Dict[str, Any]] = None

class AlertResponse(AgentResponse):
    """Alert Agent Response Model"""
    alert_type: str = Field(..., min_length=1)
    severity: SeverityLevel
    affected_systems: List[str] = Field(default_factory=list)
    immediate_actions: List[str] = Field(default_factory=list)
    escalation_required: bool = False

# Database Query Models
class QueryFilter(BaseModel):
    """Database Query Filter Model"""
    field: str = Field(..., min_length=1)
    operator: str = Field(..., min_length=1)  # "eq", "gt", "lt", "gte", "lte", "in", "regex"
    value: Union[str, int, float, bool, List[Any]]
    
class SortCriteria(BaseModel):
    """Database Sort Criteria Model"""
    field: str = Field(..., min_length=1)
    direction: str = Field("asc", pattern="^(asc|desc)$")

class PaginationParams(BaseModel):
    """Pagination Parameters Model"""
    page: int = Field(1, ge=1)
    limit: int = Field(100, ge=1, le=1000)
    sort: Optional[List[SortCriteria]] = None
    filters: Optional[List[QueryFilter]] = None

# Analytics Models
class TimeSeriesData(BaseModel):
    """Time Series Data Model"""
    timestamp: datetime
    value: float
    metadata: Optional[Dict[str, Any]] = None

class AggregationResult(BaseModel):
    """Database Aggregation Result Model"""
    field: str
    operation: str  # "sum", "avg", "min", "max", "count"
    value: Union[float, int]
    group_by: Optional[Dict[str, Any]] = None

class RiskAssessment(BaseModel):
    """Risk Assessment Model"""
    farm_id: str = Field(..., min_length=1)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    overall_risk_score: float = Field(..., ge=0, le=1)
    risk_factors: Dict[str, float] = Field(default_factory=dict)
    risk_level: SeverityLevel
    mitigation_strategies: List[str] = Field(default_factory=list)
    monitoring_recommendations: List[str] = Field(default_factory=list)
    
    class Config:
        use_enum_values = True

# Export all models
__all__ = [
    # Enums
    "SensorType", "AnomalyType", "SeverityLevel", "SupplyChainStatus", 
    "TransportationMode", "MarketType", "CropType",
    
    # Base Models
    "Location", "CostAnalysis", "SpectralBands",
    
    # Data Models
    "IoTSensorData", "SatelliteData", "SupplyChainData", 
    "FinancialData", "AnomalyData",
    
    # Response Models
    "AgentResponse", "PredictionResponse", "RecommendationResponse", "AlertResponse",
    
    # Query Models
    "QueryFilter", "SortCriteria", "PaginationParams",
    
    # Analytics Models
    "TimeSeriesData", "AggregationResult", "RiskAssessment"
]
