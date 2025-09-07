"""
Data Layer Initialization

Main initialization script for the data layer that sets up all components.
"""

import os
import sys
import logging
from pathlib import Path

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from data_layer.config import DatabaseConfig, ETLConfig, ValidationConfig
from data_layer.database.mongodb.mongodb_setup import MongoDBManager
from data_layer.ingestion.etl.data_ingestion import DataIngestionPipeline
from data_layer.utils import DataValidator, QualityMetrics

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLayerManager:
    """Main data layer manager that orchestrates all data operations"""
    
    def __init__(self):
        self.config = {
            "database": DatabaseConfig.from_env(),
            "etl": ETLConfig(),
            "validation": ValidationConfig()
        }
        
        self.mongodb_manager = None
        self.data_validator = None
        self.quality_metrics = None
        self.etl_pipeline = None
        
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize all data layer components"""
        try:
            # Initialize MongoDB manager
            self.mongodb_manager = MongoDBManager(
                host=self.config["database"].mongodb_host,
                port=self.config["database"].mongodb_port,
                database=self.config["database"].mongodb_database,
                username=self.config["database"].mongodb_username,
                password=self.config["database"].mongodb_password
            )
            
            # Initialize data validator
            self.data_validator = DataValidator(self.config["validation"])
            
            # Initialize quality metrics
            self.quality_metrics = QualityMetrics()
            
            # Initialize ETL pipeline
            self.etl_pipeline = DataIngestionPipeline(self.mongodb_manager)
            
            logger.info("✅ Data layer components initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize data layer components: {e}")
            raise
    
    def setup_database(self) -> bool:
        """Setup database with collections and indexes"""
        try:
            if not self.mongodb_manager.connect():
                logger.error("❌ Failed to connect to MongoDB")
                return False
            
            # Create collections
            self.mongodb_manager.create_collections()
            
            # Create indexes
            self.mongodb_manager.create_indexes()
            
            logger.info("✅ Database setup completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Database setup failed: {e}")
            return False
    
    def ingest_enterprise_data(self, data_directory: str) -> dict:
        """Ingest enterprise data from CSV files"""
        try:
            if not self.mongodb_manager or not self.mongodb_manager.is_connected():
                logger.error("❌ MongoDB not connected")
                return {"error": "MongoDB not connected"}
            
            # Run data ingestion
            results = self.etl_pipeline.ingest_data(data_directory)
            
            logger.info("✅ Enterprise data ingestion completed")
            return results
            
        except Exception as e:
            logger.error(f"❌ Data ingestion failed: {e}")
            return {"error": str(e)}
    
    def validate_data_quality(self, data_type: str, df) -> dict:
        """Validate data quality for specific dataset"""
        try:
            # Validate data
            valid_df, invalid_records = self.data_validator.validate_dataframe(df, data_type)
            
            # Calculate quality metrics
            quality_report = self.quality_metrics.calculate_overall_quality_score(valid_df, data_type)
            
            # Get validation report
            validation_report = self.data_validator.get_validation_report()
            
            return {
                "validation": validation_report,
                "quality": quality_report,
                "valid_records": len(valid_df),
                "invalid_records": len(invalid_records)
            }
            
        except Exception as e:
            logger.error(f"❌ Data validation failed: {e}")
            return {"error": str(e)}
    
    def get_database_stats(self) -> dict:
        """Get database statistics"""
        try:
            if not self.mongodb_manager or not self.mongodb_manager.is_connected():
                return {"error": "MongoDB not connected"}
            
            return self.mongodb_manager.get_collection_stats()
            
        except Exception as e:
            logger.error(f"❌ Failed to get database stats: {e}")
            return {"error": str(e)}
    
    def close_connections(self):
        """Close all database connections"""
        try:
            if self.mongodb_manager:
                self.mongodb_manager.close_connection()
            logger.info("✅ Database connections closed")
        except Exception as e:
            logger.error(f"❌ Error closing connections: {e}")


def main():
    """Main function to initialize and setup data layer"""
    logger.info("🚀 Initializing Organic Agriculture Data Layer")
    
    # Initialize data layer manager
    data_manager = DataLayerManager()
    
    try:
        # Setup database
        if not data_manager.setup_database():
            logger.error("❌ Database setup failed")
            return
        
        # Get database statistics
        stats = data_manager.get_database_stats()
        logger.info("📊 Database Statistics:")
        for collection, stat in stats.items():
            if 'error' not in stat:
                logger.info(f"  {collection}: {stat['count']:,} records ({stat['size_mb']:.2f} MB)")
            else:
                logger.error(f"  {collection}: {stat['error']}")
        
        logger.info("✅ Data layer initialization completed successfully")
        
    except Exception as e:
        logger.error(f"❌ Data layer initialization failed: {e}")
    
    finally:
        data_manager.close_connections()


if __name__ == "__main__":
    main()
