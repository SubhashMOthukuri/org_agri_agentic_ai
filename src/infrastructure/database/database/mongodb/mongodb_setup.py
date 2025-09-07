"""
MongoDB Setup and Configuration for Organic Agriculture Agentic AI
Handles connection, data models, and data ingestion for all enterprise datasets
"""

import os
import sys
import json
import pandas as pd
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.errors import ConnectionFailure, DuplicateKeyError
import logging
from dotenv import load_dotenv

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

# Load environment variables from main project folder
load_dotenv(os.path.join(project_root, '.env'))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MongoDBManager:
    """MongoDB Manager for Organic Agriculture Agentic AI System"""
    
    def __init__(self, connection_string: str = None):
        """
        Initialize MongoDB connection
        
        Args:
            connection_string: MongoDB connection string (defaults to localhost)
        """
        self.connection_string = connection_string or os.getenv(
            'MONGODB_URI', 
            'mongodb://localhost:27017/'
        )
        self.database_name = 'organic_agriculture_ai'
        self.client = None
        self.db = None
        
    def connect(self) -> bool:
        """Establish connection to MongoDB"""
        try:
            self.client = MongoClient(self.connection_string, serverSelectionTimeoutMS=5000)
            # Test connection
            self.client.admin.command('ping')
            self.db = self.client[self.database_name]
            logger.info(f"✅ Connected to MongoDB: {self.database_name}")
            return True
        except ConnectionFailure as e:
            logger.error(f"❌ Failed to connect to MongoDB: {e}")
            return False
    
    def create_indexes(self):
        """Create optimized indexes for all collections"""
        try:
            # IoT Sensor Data indexes
            self.db.iot_sensor_data.create_index([
                ("timestamp", ASCENDING),
                ("sensor_id", ASCENDING)
            ])
            self.db.iot_sensor_data.create_index([
                ("farm_id", ASCENDING),
                ("sensor_type", ASCENDING)
            ])
            
            # Satellite Data indexes
            self.db.satellite_data.create_index([
                ("timestamp", ASCENDING),
                ("farm_id", ASCENDING)
            ])
            self.db.satellite_data.create_index([
                ("vegetation_index", ASCENDING),
                ("timestamp", ASCENDING)
            ])
            
            # Supply Chain Data indexes
            self.db.supply_chain_data.create_index([
                ("timestamp", ASCENDING),
                ("farm_id", ASCENDING)
            ])
            self.db.supply_chain_data.create_index([
                ("supply_chain_id", ASCENDING),
                ("status", ASCENDING)
            ])
            
            # Financial Data indexes
            self.db.financial_data.create_index([
                ("timestamp", ASCENDING),
                ("farm_id", ASCENDING)
            ])
            self.db.financial_data.create_index([
                ("crop_type", ASCENDING),
                ("market_type", ASCENDING)
            ])
            
            # Anomaly Data indexes
            self.db.anomaly_data.create_index([
                ("timestamp", ASCENDING),
                ("anomaly_type", ASCENDING)
            ])
            self.db.anomaly_data.create_index([
                ("farm_id", ASCENDING),
                ("severity", ASCENDING)
            ])
            
            logger.info("✅ Created optimized indexes for all collections")
            
        except Exception as e:
            logger.error(f"❌ Error creating indexes: {e}")
    
    def create_collections(self):
        """Create collections with validation schemas"""
        try:
            # IoT Sensor Data Collection
            self.db.create_collection("iot_sensor_data", validator={
                "$jsonSchema": {
                    "bsonType": "object",
                    "required": ["sensor_id", "farm_id", "timestamp", "sensor_type", "value"],
                    "properties": {
                        "sensor_id": {"bsonType": "string"},
                        "farm_id": {"bsonType": "string"},
                        "timestamp": {"bsonType": "date"},
                        "sensor_type": {"bsonType": "string"},
                        "value": {"bsonType": "number"},
                        "unit": {"bsonType": "string"},
                        "location": {
                            "bsonType": "object",
                            "properties": {
                                "latitude": {"bsonType": "number"},
                                "longitude": {"bsonType": "number"}
                            }
                        }
                    }
                }
            })
            
            # Satellite Data Collection
            self.db.create_collection("satellite_data", validator={
                "$jsonSchema": {
                    "bsonType": "object",
                    "required": ["farm_id", "timestamp", "vegetation_index"],
                    "properties": {
                        "farm_id": {"bsonType": "string"},
                        "timestamp": {"bsonType": "date"},
                        "vegetation_index": {"bsonType": "number"},
                        "spectral_bands": {"bsonType": "object"},
                        "cloud_cover": {"bsonType": "number"},
                        "quality_score": {"bsonType": "number"}
                    }
                }
            })
            
            # Supply Chain Data Collection
            self.db.create_collection("supply_chain_data", validator={
                "$jsonSchema": {
                    "bsonType": "object",
                    "required": ["supply_chain_id", "farm_id", "timestamp", "status"],
                    "properties": {
                        "supply_chain_id": {"bsonType": "string"},
                        "farm_id": {"bsonType": "string"},
                        "timestamp": {"bsonType": "date"},
                        "status": {"bsonType": "string"},
                        "transportation_mode": {"bsonType": "string"},
                        "estimated_delivery": {"bsonType": "date"},
                        "risk_score": {"bsonType": "number"}
                    }
                }
            })
            
            # Financial Data Collection
            self.db.create_collection("financial_data", validator={
                "$jsonSchema": {
                    "bsonType": "object",
                    "required": ["farm_id", "timestamp", "crop_type", "market_price"],
                    "properties": {
                        "farm_id": {"bsonType": "string"},
                        "timestamp": {"bsonType": "date"},
                        "crop_type": {"bsonType": "string"},
                        "market_price": {"bsonType": "number"},
                        "market_type": {"bsonType": "string"},
                        "cost_analysis": {"bsonType": "object"},
                        "profit_margin": {"bsonType": "number"}
                    }
                }
            })
            
            # Anomaly Data Collection
            self.db.create_collection("anomaly_data", validator={
                "$jsonSchema": {
                    "bsonType": "object",
                    "required": ["farm_id", "timestamp", "anomaly_type", "severity"],
                    "properties": {
                        "anomaly_id": {"bsonType": "string"},
                        "farm_id": {"bsonType": "string"},
                        "timestamp": {"bsonType": "date"},
                        "anomaly_type": {"bsonType": "string"},
                        "severity": {"bsonType": "string"},
                        "description": {"bsonType": "string"},
                        "affected_metrics": {"bsonType": "array"},
                        "recommended_actions": {"bsonType": "array"}
                    }
                }
            })
            
            logger.info("✅ Created collections with validation schemas")
            
        except Exception as e:
            logger.error(f"❌ Error creating collections: {e}")
    
    def ingest_iot_data(self, csv_path: str) -> int:
        """Ingest IoT sensor data from CSV"""
        try:
            df = pd.read_csv(csv_path)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Convert to list of dictionaries
            records = df.to_dict('records')
            
            # Insert in batches for better performance
            batch_size = 1000
            inserted_count = 0
            
            for i in range(0, len(records), batch_size):
                batch = records[i:i + batch_size]
                result = self.db.iot_sensor_data.insert_many(batch, ordered=False)
                inserted_count += len(result.inserted_ids)
                
            logger.info(f"✅ Ingested {inserted_count} IoT sensor records")
            return inserted_count
            
        except Exception as e:
            logger.error(f"❌ Error ingesting IoT data: {e}")
            return 0
    
    def ingest_satellite_data(self, csv_path: str) -> int:
        """Ingest satellite data from CSV"""
        try:
            df = pd.read_csv(csv_path)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            records = df.to_dict('records')
            result = self.db.satellite_data.insert_many(records, ordered=False)
            
            logger.info(f"✅ Ingested {len(result.inserted_ids)} satellite records")
            return len(result.inserted_ids)
            
        except Exception as e:
            logger.error(f"❌ Error ingesting satellite data: {e}")
            return 0
    
    def ingest_supply_chain_data(self, csv_path: str) -> int:
        """Ingest supply chain data from CSV"""
        try:
            df = pd.read_csv(csv_path)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['estimated_delivery'] = pd.to_datetime(df['estimated_delivery'])
            
            records = df.to_dict('records')
            result = self.db.supply_chain_data.insert_many(records, ordered=False)
            
            logger.info(f"✅ Ingested {len(result.inserted_ids)} supply chain records")
            return len(result.inserted_ids)
            
        except Exception as e:
            logger.error(f"❌ Error ingesting supply chain data: {e}")
            return 0
    
    def ingest_financial_data(self, csv_path: str) -> int:
        """Ingest financial data from CSV"""
        try:
            df = pd.read_csv(csv_path)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            records = df.to_dict('records')
            result = self.db.financial_data.insert_many(records, ordered=False)
            
            logger.info(f"✅ Ingested {len(result.inserted_ids)} financial records")
            return len(result.inserted_ids)
            
        except Exception as e:
            logger.error(f"❌ Error ingesting financial data: {e}")
            return 0
    
    def ingest_anomaly_data(self, csv_path: str) -> int:
        """Ingest anomaly data from CSV"""
        try:
            df = pd.read_csv(csv_path)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            records = df.to_dict('records')
            result = self.db.anomaly_data.insert_many(records, ordered=False)
            
            logger.info(f"✅ Ingested {len(result.inserted_ids)} anomaly records")
            return len(result.inserted_ids)
            
        except Exception as e:
            logger.error(f"❌ Error ingesting anomaly data: {e}")
            return 0
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics for all collections"""
        stats = {}
        
        collections = [
            'iot_sensor_data',
            'satellite_data', 
            'supply_chain_data',
            'financial_data',
            'anomaly_data'
        ]
        
        for collection_name in collections:
            try:
                collection = self.db[collection_name]
                count = collection.count_documents({})
                stats[collection_name] = {
                    'count': count,
                    'size_mb': self.db.command("collStats", collection_name).get('size', 0) / (1024 * 1024)
                }
            except Exception as e:
                stats[collection_name] = {'error': str(e)}
        
        return stats
    
    def close_connection(self):
        """Close MongoDB connection"""
        if self.client:
            self.client.close()
            logger.info("✅ MongoDB connection closed")

def main():
    """Main function to set up MongoDB and ingest all data"""
    logger.info("🚀 Starting MongoDB setup for Organic Agriculture Agentic AI")
    
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
        
        # Data paths
        data_dir = "/Users/subhashmothukurigmail.com/Projects/org_agri/DeepAgentPrototype/data/data/enterprise"
        
        # Ingest all enterprise data
        logger.info("📊 Starting data ingestion...")
        
        total_records = 0
        
        # IoT Sensor Data (if file exists and is not empty)
        iot_path = os.path.join(data_dir, "iot_sensor_data.csv")
        if os.path.exists(iot_path) and os.path.getsize(iot_path) > 0:
            total_records += mongo_manager.ingest_iot_data(iot_path)
        else:
            logger.warning("⚠️ IoT sensor data file not found or empty")
        
        # Satellite Data
        satellite_path = os.path.join(data_dir, "satellite_data.csv")
        if os.path.exists(satellite_path):
            total_records += mongo_manager.ingest_satellite_data(satellite_path)
        
        # Supply Chain Data
        supply_chain_path = os.path.join(data_dir, "supply_chain_data.csv")
        if os.path.exists(supply_chain_path):
            total_records += mongo_manager.ingest_supply_chain_data(supply_chain_path)
        
        # Financial Data
        financial_path = os.path.join(data_dir, "financial_data.csv")
        if os.path.exists(financial_path):
            total_records += mongo_manager.ingest_financial_data(financial_path)
        
        # Anomaly Data
        anomaly_path = os.path.join(data_dir, "anomaly_data.csv")
        if os.path.exists(anomaly_path):
            total_records += mongo_manager.ingest_anomaly_data(anomaly_path)
        
        # Get final statistics
        stats = mongo_manager.get_collection_stats()
        
        logger.info("🎉 MongoDB setup completed successfully!")
        logger.info(f"📊 Total records ingested: {total_records:,}")
        
        # Print collection statistics
        for collection, stat in stats.items():
            if 'error' not in stat:
                logger.info(f"📈 {collection}: {stat['count']:,} records ({stat['size_mb']:.2f} MB)")
            else:
                logger.error(f"❌ {collection}: {stat['error']}")
        
    except Exception as e:
        logger.error(f"❌ Error during MongoDB setup: {e}")
    
    finally:
        mongo_manager.close_connection()

if __name__ == "__main__":
    main()
