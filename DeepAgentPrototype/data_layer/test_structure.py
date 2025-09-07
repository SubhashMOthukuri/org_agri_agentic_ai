#!/usr/bin/env python3
"""
Test script to verify data layer structure and imports
"""

import sys
import os

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

def test_imports():
    """Test all data layer imports"""
    try:
        print("🧪 Testing data layer imports...")
        
        # Test main data layer import
        from DeepAgentPrototype.data_layer import DataLayerManager
        print("✅ DataLayerManager import successful")
        
        # Test configuration imports
        from DeepAgentPrototype.data_layer.config import DatabaseConfig, ETLConfig, ValidationConfig
        print("✅ Configuration imports successful")
        
        # Test model imports
        from DeepAgentPrototype.data_layer.models import IoTSensorData, SensorType
        print("✅ Model imports successful")
        
        # Test utility imports
        from DeepAgentPrototype.data_layer.utils import DataValidator, QualityMetrics
        print("✅ Utility imports successful")
        
        # Test database imports
        from DeepAgentPrototype.data_layer.database.mongodb.mongodb_setup import MongoDBManager
        print("✅ Database imports successful")
        
        # Test ETL imports
        from DeepAgentPrototype.data_layer.ingestion.etl.data_ingestion import DataIngestionPipeline
        print("✅ ETL imports successful")
        
        print("\n🎉 All imports successful! Data layer structure is working correctly.")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_configuration():
    """Test configuration creation"""
    try:
        print("\n🧪 Testing configuration creation...")
        
        from DeepAgentPrototype.data_layer.config import DatabaseConfig, ETLConfig, ValidationConfig
        
        # Test database config
        db_config = DatabaseConfig.from_env()
        print(f"✅ Database config created: {db_config.mongodb_host}:{db_config.mongodb_port}")
        
        # Test ETL config
        etl_config = ETLConfig()
        print(f"✅ ETL config created: batch_size={etl_config.batch_size}")
        
        # Test validation config
        val_config = ValidationConfig()
        print(f"✅ Validation config created: validation_enabled={val_config.enable_validation}")
        
        print("🎉 Configuration creation successful!")
        return True
        
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return False

def test_models():
    """Test model creation"""
    try:
        print("\n🧪 Testing model creation...")
        
        from DeepAgentPrototype.data_layer.models import IoTSensorData, SensorType
        from datetime import datetime
        
        # Test IoT sensor data model
        sensor_data = IoTSensorData(
            sensor_id="sensor_001",
            farm_id="farm_001",
            timestamp=datetime.utcnow(),
            sensor_type=SensorType.TEMPERATURE,
            value=25.5,
            unit="celsius"
        )
        print(f"✅ IoT sensor data model created: {sensor_data.sensor_type}")
        
        print("🎉 Model creation successful!")
        return True
        
    except Exception as e:
        print(f"❌ Model error: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 Testing Data Layer Structure")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_configuration,
        test_models
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Data layer structure is ready for use.")
        return True
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
