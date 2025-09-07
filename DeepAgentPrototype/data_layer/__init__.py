"""
Data Layer Package for Organic Agriculture Agentic AI

This package provides comprehensive data management capabilities including:
- Database connections and management
- Data models and schemas
- ETL pipelines and data ingestion
- Data validation and quality assurance
- Configuration and utilities

Author: Organic Agriculture AI Team
Version: 1.0.0
"""

from .database.mongodb.mongodb_setup import MongoDBManager
from .models import (
    IoTSensorData, SatelliteData, SupplyChainData, 
    FinancialData, AnomalyData, SensorType, AnomalyType,
    SeverityLevel, SupplyChainStatus, TransportationMode,
    MarketType, CropType, Location, SpectralBands, CostAnalysis
)
from .ingestion.etl.data_ingestion import DataIngestionPipeline

__version__ = "1.0.0"
__author__ = "Organic Agriculture AI Team"

__all__ = [
    # Database
    "MongoDBManager",
    
    # Models
    "IoTSensorData", "SatelliteData", "SupplyChainData", 
    "FinancialData", "AnomalyData",
    
    # Enums
    "SensorType", "AnomalyType", "SeverityLevel", 
    "SupplyChainStatus", "TransportationMode", "MarketType", "CropType",
    
    # Base Models
    "Location", "SpectralBands", "CostAnalysis",
    
    # ETL
    "DataIngestionPipeline"
]
