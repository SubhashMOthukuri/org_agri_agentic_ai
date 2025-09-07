"""
Data Ingestion Pipeline for Organic Agriculture Agentic AI
Handles CSV to MongoDB conversion with data validation and transformation
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
import logging
from pathlib import Path
import json

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
sys.path.insert(0, project_root)

from DeepAgentPrototype.data_layer.database.mongodb.mongodb_setup import MongoDBManager
from DeepAgentPrototype.data_layer.models.models import (
    IoTSensorData, SatelliteData, SupplyChainData, 
    FinancialData, AnomalyData, SensorType, AnomalyType, 
    SeverityLevel, SupplyChainStatus, TransportationMode, 
    MarketType, CropType, Location, SpectralBands, CostAnalysis
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataIngestionPipeline:
    """Data Ingestion Pipeline for converting CSV data to MongoDB documents"""
    
    def __init__(self, mongo_manager: MongoDBManager):
        self.mongo_manager = mongo_manager
        
    def transform_iot_data(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Transform IoT sensor data for MongoDB ingestion"""
        try:
            records = []
            
            for _, row in df.iterrows():
                # Map sensor types
                sensor_type_mapping = {
                    'temperature': SensorType.TEMPERATURE,
                    'humidity': SensorType.HUMIDITY,
                    'soil_moisture': SensorType.SOIL_MOISTURE,
                    'ph': SensorType.PH_LEVEL,
                    'light': SensorType.LIGHT_INTENSITY,
                    'co2': SensorType.CO2_LEVEL,
                    'nitrogen': SensorType.NITROGEN_LEVEL,
                    'phosphorus': SensorType.PHOSPHORUS_LEVEL,
                    'potassium': SensorType.POTASSIUM_LEVEL
                }
                
                sensor_type = sensor_type_mapping.get(
                    row.get('sensor_type', '').lower(), 
                    SensorType.TEMPERATURE
                )
                
                record = {
                    'sensor_id': str(row.get('sensor_id', '')),
                    'farm_id': str(row.get('farm_id', '')),
                    'timestamp': pd.to_datetime(row.get('timestamp', datetime.utcnow())),
                    'sensor_type': sensor_type.value,
                    'value': float(row.get('value', 0)),
                    'unit': str(row.get('unit', '')),
                    'quality_score': float(row.get('quality_score', 1.0)),
                    'battery_level': float(row.get('battery_level', 100)) if pd.notna(row.get('battery_level')) else None,
                    'signal_strength': float(row.get('signal_strength', 100)) if pd.notna(row.get('signal_strength')) else None
                }
                
                # Add location if available
                if pd.notna(row.get('latitude')) and pd.notna(row.get('longitude')):
                    record['location'] = {
                        'latitude': float(row['latitude']),
                        'longitude': float(row['longitude']),
                        'altitude': float(row.get('altitude', 0)) if pd.notna(row.get('altitude')) else None
                    }
                
                records.append(record)
            
            logger.info(f"✅ Transformed {len(records)} IoT sensor records")
            return records
            
        except Exception as e:
            logger.error(f"❌ Error transforming IoT data: {e}")
            return []
    
    def transform_satellite_data(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Transform satellite data for MongoDB ingestion"""
        try:
            records = []
            
            for _, row in df.iterrows():
                # Create spectral bands if available - handle different column naming
                spectral_bands = None
                if all(pd.notna(row.get(band)) for band in ['red_band', 'green_band', 'blue_band', 'nir_band', 'swir1_band', 'swir2_band']):
                    spectral_bands = {
                        'red': float(row['red_band']),
                        'green': float(row['green_band']),
                        'blue': float(row['blue_band']),
                        'nir': float(row['nir_band']),
                        'swir1': float(row['swir1_band']),
                        'swir2': float(row['swir2_band'])
                    }
                elif all(pd.notna(row.get(band)) for band in ['red', 'green', 'blue', 'nir', 'swir1', 'swir2']):
                    spectral_bands = {
                        'red': float(row['red']),
                        'green': float(row['green']),
                        'blue': float(row['blue']),
                        'nir': float(row['nir']),
                        'swir1': float(row['swir1']),
                        'swir2': float(row['swir2'])
                    }
                else:
                    # Create default spectral bands if none available
                    spectral_bands = {
                        'red': 0.0,
                        'green': 0.0,
                        'blue': 0.0,
                        'nir': 0.0,
                        'swir1': 0.0,
                        'swir2': 0.0
                    }
                
                record = {
                    'farm_id': str(row.get('farm_id', '')),
                    'timestamp': pd.to_datetime(row.get('timestamp', datetime.utcnow())),
                    'vegetation_index': float(row.get('ndvi', row.get('vegetation_index', 0))),
                    'spectral_bands': spectral_bands,
                    'cloud_cover': float(row.get('cloud_cover', 0)),
                    'quality_score': float(row.get('image_quality', row.get('quality_score', 1.0))),
                    'satellite_name': str(row.get('satellite_name', '')) if pd.notna(row.get('satellite_name')) else None,
                    'resolution': float(row.get('resolution', 0)) if pd.notna(row.get('resolution')) else None
                }
                
                records.append(record)
            
            logger.info(f"✅ Transformed {len(records)} satellite records")
            return records
            
        except Exception as e:
            logger.error(f"❌ Error transforming satellite data: {e}")
            return []
    
    def transform_supply_chain_data(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Transform supply chain data for MongoDB ingestion"""
        try:
            records = []
            
            for _, row in df.iterrows():
                # Map transportation modes
                transport_mapping = {
                    'truck': TransportationMode.TRUCK,
                    'train': TransportationMode.TRAIN,
                    'ship': TransportationMode.SHIP,
                    'airplane': TransportationMode.AIRPLANE,
                    'drone': TransportationMode.DRONE
                }
                
                transport_mode = transport_mapping.get(
                    row.get('transportation_mode', '').lower(),
                    TransportationMode.TRUCK
                )
                
                # Map status
                status_mapping = {
                    'pending': SupplyChainStatus.PENDING,
                    'in_transit': SupplyChainStatus.IN_TRANSIT,
                    'delivered': SupplyChainStatus.DELIVERED,
                    'delayed': SupplyChainStatus.DELAYED,
                    'cancelled': SupplyChainStatus.CANCELLED
                }
                
                status = status_mapping.get(
                    row.get('status', '').lower(),
                    SupplyChainStatus.PENDING
                )
                
                record = {
                    'supply_chain_id': str(row.get('supply_chain_id', '')),
                    'farm_id': str(row.get('farm_id', '')),
                    'timestamp': pd.to_datetime(row.get('timestamp', datetime.utcnow())),
                    'status': status.value,
                    'transportation_mode': transport_mode.value,
                    'estimated_delivery': pd.to_datetime(row.get('estimated_delivery', datetime.utcnow())),
                    'actual_delivery': pd.to_datetime(row.get('actual_delivery')) if pd.notna(row.get('actual_delivery')) else None,
                    'risk_score': float(row.get('risk_score', 0)),
                    'cargo_type': str(row.get('cargo_type', '')),
                    'cargo_weight': float(row.get('cargo_weight', 0)) if pd.notna(row.get('cargo_weight')) else None,
                    'cargo_volume': float(row.get('cargo_volume', 0)) if pd.notna(row.get('cargo_volume')) else None,
                    'temperature_controlled': bool(row.get('temperature_controlled', False)),
                    'special_requirements': str(row.get('special_requirements', '')).split(',') if pd.notna(row.get('special_requirements')) else None
                }
                
                # Add locations if available
                if pd.notna(row.get('origin_lat')) and pd.notna(row.get('origin_lon')):
                    record['origin_location'] = {
                        'latitude': float(row['origin_lat']),
                        'longitude': float(row['origin_lon'])
                    }
                
                if pd.notna(row.get('dest_lat')) and pd.notna(row.get('dest_lon')):
                    record['destination_location'] = {
                        'latitude': float(row['dest_lat']),
                        'longitude': float(row['dest_lon'])
                    }
                
                records.append(record)
            
            logger.info(f"✅ Transformed {len(records)} supply chain records")
            return records
            
        except Exception as e:
            logger.error(f"❌ Error transforming supply chain data: {e}")
            return []
    
    def transform_financial_data(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Transform financial data for MongoDB ingestion"""
        try:
            records = []
            
            for _, row in df.iterrows():
                # Map crop types
                crop_mapping = {
                    'tomato': CropType.TOMATO,
                    'lettuce': CropType.LETTUCE,
                    'carrot': CropType.CARROT,
                    'potato': CropType.POTATO,
                    'wheat': CropType.WHEAT,
                    'corn': CropType.CORN,
                    'soybean': CropType.SOYBEAN,
                    'rice': CropType.RICE,
                    'spinach': CropType.SPINACH,
                    'cucumber': CropType.CUCUMBER
                }
                
                crop_type = crop_mapping.get(
                    row.get('crop_type', '').lower(),
                    CropType.TOMATO
                )
                
                # Map market types
                market_mapping = {
                    'local': MarketType.LOCAL,
                    'regional': MarketType.REGIONAL,
                    'national': MarketType.NATIONAL,
                    'international': MarketType.INTERNATIONAL,
                    'organic_certified': MarketType.ORGANIC_CERTIFIED
                }
                
                market_type = market_mapping.get(
                    row.get('market_type', '').lower(),
                    MarketType.LOCAL
                )
                
                # Create cost analysis if available
                cost_analysis = None
                if all(pd.notna(row.get(cost)) for cost in ['production_cost', 'transportation_cost', 'storage_cost', 'marketing_cost']):
                    cost_analysis = {
                        'production_cost': float(row['production_cost']),
                        'transportation_cost': float(row['transportation_cost']),
                        'storage_cost': float(row['storage_cost']),
                        'marketing_cost': float(row['marketing_cost']),
                        'total_cost': float(row.get('total_cost', 0))
                    }
                
                record = {
                    'farm_id': str(row.get('farm_id', '')),
                    'timestamp': pd.to_datetime(row.get('timestamp', datetime.utcnow())),
                    'crop_type': crop_type.value,
                    'market_price': float(row.get('market_price', 0)),
                    'market_type': market_type.value,
                    'cost_analysis': cost_analysis,
                    'profit_margin': float(row.get('profit_margin', 0)) if pd.notna(row.get('profit_margin')) else None,
                    'demand_forecast': float(row.get('demand_forecast', 0)) if pd.notna(row.get('demand_forecast')) else None,
                    'supply_forecast': float(row.get('supply_forecast', 0)) if pd.notna(row.get('supply_forecast')) else None,
                    'price_volatility': float(row.get('price_volatility', 0)) if pd.notna(row.get('price_volatility')) else None,
                    'market_trend': str(row.get('market_trend', '')) if pd.notna(row.get('market_trend')) else None
                }
                
                records.append(record)
            
            logger.info(f"✅ Transformed {len(records)} financial records")
            return records
            
        except Exception as e:
            logger.error(f"❌ Error transforming financial data: {e}")
            return []
    
    def transform_anomaly_data(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Transform anomaly data for MongoDB ingestion"""
        try:
            records = []
            
            for idx, row in df.iterrows():
                # Map anomaly types
                anomaly_mapping = {
                    'sensor_malfunction': AnomalyType.SENSOR_MALFUNCTION,
                    'pest_outbreak': AnomalyType.PEST_OUTBREAK,
                    'disease_detection': AnomalyType.DISEASE_DETECTION,
                    'weather_extreme': AnomalyType.WEATHER_EXTREME,
                    'soil_contamination': AnomalyType.SOIL_CONTAMINATION,
                    'water_shortage': AnomalyType.WATER_SHORTAGE,
                    'equipment_failure': AnomalyType.EQUIPMENT_FAILURE,
                    'supply_chain_delay': AnomalyType.SUPPLY_CHAIN_DELAY,
                    'regulatory_changes': AnomalyType.SENSOR_MALFUNCTION,  # Map to closest match
                    'supply_chain_disruption': AnomalyType.SUPPLY_CHAIN_DELAY,
                    'market_crash': AnomalyType.SENSOR_MALFUNCTION,  # Map to closest match
                    'extreme_weather': AnomalyType.WEATHER_EXTREME,
                    'disease_outbreak': AnomalyType.DISEASE_DETECTION
                }
                
                anomaly_type = anomaly_mapping.get(
                    row.get('anomaly_type', '').lower(),
                    AnomalyType.SENSOR_MALFUNCTION
                )
                
                # Map severity levels
                severity_mapping = {
                    'low': SeverityLevel.LOW,
                    'medium': SeverityLevel.MEDIUM,
                    'high': SeverityLevel.HIGH,
                    'critical': SeverityLevel.CRITICAL
                }
                
                severity = severity_mapping.get(
                    row.get('severity', '').lower(),
                    SeverityLevel.MEDIUM
                )
                
                # Generate anomaly_id if not present
                anomaly_id = str(row.get('anomaly_id', f'anomaly_{idx}_{int(datetime.utcnow().timestamp())}'))
                
                record = {
                    'anomaly_id': anomaly_id,
                    'farm_id': str(row.get('farm_id', '')),
                    'timestamp': pd.to_datetime(row.get('timestamp', datetime.utcnow())),
                    'anomaly_type': anomaly_type.value,
                    'severity': severity.value,
                    'description': str(row.get('description', '')),
                    'affected_metrics': str(row.get('affected_metrics', '')).split(',') if pd.notna(row.get('affected_metrics')) else [],
                    'recommended_actions': str(row.get('recommended_actions', '')).split(',') if pd.notna(row.get('recommended_actions')) else [],
                    'confidence_score': float(row.get('confidence_score', 1.0)),
                    'resolved': bool(row.get('resolved', False)),
                    'resolution_timestamp': pd.to_datetime(row.get('resolution_timestamp')) if pd.notna(row.get('resolution_timestamp')) else None,
                    'resolution_notes': str(row.get('resolution_notes', '')) if pd.notna(row.get('resolution_notes')) else None
                }
                
                records.append(record)
            
            logger.info(f"✅ Transformed {len(records)} anomaly records")
            return records
            
        except Exception as e:
            logger.error(f"❌ Error transforming anomaly data: {e}")
            return []
    
    def ingest_data(self, data_dir: str) -> Dict[str, int]:
        """Ingest all data from CSV files"""
        results = {}
        
        try:
            # IoT Sensor Data
            iot_path = os.path.join(data_dir, "iot_sensor_data.csv")
            if os.path.exists(iot_path) and os.path.getsize(iot_path) > 0:
                logger.info("📊 Processing IoT sensor data... (adding next 1M records)") 
                df = pd.read_csv(iot_path)
                
                # Process next 1 million records (skip first 1M, take next 1M)
                original_count = len(df)
                if original_count > 2000000:
                    df = df.iloc[1000000:2000000]  # Skip first 1M, take next 1M
                    logger.info(f"📊 Processing next 1M records (records 1,000,001 to 2,000,000 from {original_count:,} total)")
                elif original_count > 1000000:
                    df = df.iloc[1000000:]  # Take remaining records after first 1M
                    logger.info(f"📊 Processing remaining {len(df):,} records after first 1M (from {original_count:,} total)")
                else:
                    logger.info(f"📊 Processing all {len(df):,} records (less than 1M total)")
                
                records = self.transform_iot_data(df)
                if records:
                    # Insert in batches
                    batch_size = 1000
                    inserted_count = 0
                    for i in range(0, len(records), batch_size):
                        batch = records[i:i + batch_size]
                        result = self.mongo_manager.db.iot_sensor_data.insert_many(batch, ordered=False)
                        inserted_count += len(result.inserted_ids)
                    results['iot_sensor_data'] = inserted_count
                    logger.info(f"✅ Ingested {inserted_count} IoT sensor records")
            else:
                logger.warning("⚠️ IoT sensor data file not found or empty")
                results['iot_sensor_data'] = 0
            
            # Satellite Data
            satellite_path = os.path.join(data_dir, "satellite_data.csv")
            if os.path.exists(satellite_path):
                logger.info("📊 Processing satellite data...")
                df = pd.read_csv(satellite_path)
                records = self.transform_satellite_data(df)
                if records:
                    result = self.mongo_manager.db.satellite_data.insert_many(records, ordered=False)
                    results['satellite_data'] = len(result.inserted_ids)
                    logger.info(f"✅ Ingested {len(result.inserted_ids)} satellite records")
            
            # Supply Chain Data
            supply_chain_path = os.path.join(data_dir, "supply_chain_data.csv")
            if os.path.exists(supply_chain_path):
                logger.info("📊 Processing supply chain data...")
                df = pd.read_csv(supply_chain_path)
                records = self.transform_supply_chain_data(df)
                if records:
                    result = self.mongo_manager.db.supply_chain_data.insert_many(records, ordered=False)
                    results['supply_chain_data'] = len(result.inserted_ids)
                    logger.info(f"✅ Ingested {len(result.inserted_ids)} supply chain records")
            
            # Financial Data
            financial_path = os.path.join(data_dir, "financial_data.csv")
            if os.path.exists(financial_path):
                logger.info("📊 Processing financial data...")
                df = pd.read_csv(financial_path)
                records = self.transform_financial_data(df)
                if records:
                    result = self.mongo_manager.db.financial_data.insert_many(records, ordered=False)
                    results['financial_data'] = len(result.inserted_ids)
                    logger.info(f"✅ Ingested {len(result.inserted_ids)} financial records")
            
            # Anomaly Data
            anomaly_path = os.path.join(data_dir, "anomaly_data.csv")
            if os.path.exists(anomaly_path):
                logger.info("📊 Processing anomaly data...")
                df = pd.read_csv(anomaly_path)
                records = self.transform_anomaly_data(df)
                if records:
                    result = self.mongo_manager.db.anomaly_data.insert_many(records, ordered=False)
                    results['anomaly_data'] = len(result.inserted_ids)
                    logger.info(f"✅ Ingested {len(result.inserted_ids)} anomaly records")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Error during data ingestion: {e}")
            return results

def main():
    """Main function to run data ingestion pipeline"""
    logger.info("🚀 Starting Data Ingestion Pipeline for Organic Agriculture Agentic AI")
    
    # Initialize MongoDB manager
    mongo_manager = MongoDBManager()
    
    # Connect to MongoDB
    if not mongo_manager.connect():
        logger.error("❌ Failed to connect to MongoDB. Please ensure MongoDB is running.")
        return
    
    try:
        # Create collections and indexes
        mongo_manager.create_collections()
        mongo_manager.create_indexes()
        
        # Initialize data ingestion pipeline
        pipeline = DataIngestionPipeline(mongo_manager)
        
        # Data directory
        data_dir = "/Users/subhashmothukurigmail.com/Projects/org_agri/DeepAgentPrototype/data/data/enterprise"
        
        # Run data ingestion
        results = pipeline.ingest_data(data_dir)
        
        # Print results
        total_records = sum(results.values())
        logger.info("🎉 Data ingestion completed successfully!")
        logger.info(f"📊 Total records ingested: {total_records:,}")
        
        for collection, count in results.items():
            logger.info(f"📈 {collection}: {count:,} records")
        
        # Get final statistics
        stats = mongo_manager.get_collection_stats()
        logger.info("\n📊 Final Collection Statistics:")
        for collection, stat in stats.items():
            if 'error' not in stat:
                logger.info(f"📈 {collection}: {stat['count']:,} records ({stat['size_mb']:.2f} MB)")
            else:
                logger.error(f"❌ {collection}: {stat['error']}")
        
    except Exception as e:
        logger.error(f"❌ Error during data ingestion pipeline: {e}")
    
    finally:
        mongo_manager.close_connection()

if __name__ == "__main__":
    main()
