#!/usr/bin/env python3
"""
MongoDB Setup Runner for Organic Agriculture Agentic AI
This script sets up MongoDB and ingests all enterprise data
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

# Add the backend directory to Python path
backend_dir = Path(__file__).parent
project_root = backend_dir.parent.parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_mongodb_running():
    """Check if MongoDB is running"""
    try:
        import pymongo
        client = pymongo.MongoClient('mongodb://localhost:27017/', serverSelectionTimeoutMS=5000)
        client.admin.command('ping')
        logger.info("✅ MongoDB is running")
        return True
    except Exception as e:
        logger.error(f"❌ MongoDB is not running: {e}")
        logger.info("💡 Please start MongoDB with: brew services start mongodb-community")
        return False

def install_requirements():
    """Install required packages"""
    try:
        logger.info("📦 Installing requirements from main project...")
        # Use the main project's requirement.txt
        main_requirements = os.path.join(project_root, "requirement.txt")
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", main_requirements
        ], check=True)
        logger.info("✅ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Failed to install requirements: {e}")
        return False

def run_mongodb_setup():
    """Run MongoDB setup and data ingestion"""
    try:
        logger.info("🚀 Starting MongoDB setup...")
        
        # Import and run the setup
        from mongodb_setup import main as setup_main
        setup_main()
        
        logger.info("✅ MongoDB setup completed")
        return True
        
    except Exception as e:
        logger.error(f"❌ MongoDB setup failed: {e}")
        return False

def run_data_ingestion():
    """Run data ingestion pipeline"""
    try:
        logger.info("📊 Starting data ingestion...")
        
        # Import and run the data ingestion
        from data_ingestion import main as ingestion_main
        ingestion_main()
        
        logger.info("✅ Data ingestion completed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Data ingestion failed: {e}")
        return False

def main():
    """Main function"""
    logger.info("🌱 Organic Agriculture Agentic AI - MongoDB Setup")
    logger.info("=" * 60)
    
    # Check if MongoDB is running
    if not check_mongodb_running():
        logger.error("❌ Please start MongoDB first")
        return False
    
    # Install requirements
    if not install_requirements():
        logger.error("❌ Failed to install requirements")
        return False
    
    # Run MongoDB setup
    if not run_mongodb_setup():
        logger.error("❌ MongoDB setup failed")
        return False
    
    # Run data ingestion
    if not run_data_ingestion():
        logger.error("❌ Data ingestion failed")
        return False
    
    logger.info("🎉 MongoDB setup and data ingestion completed successfully!")
    logger.info("📊 Your enterprise data is now ready in MongoDB")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
