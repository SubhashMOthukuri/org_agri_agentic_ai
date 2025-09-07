"""
ETL Pipeline Configuration

Configuration settings for Extract, Transform, Load operations.
"""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class ETLConfig(BaseModel):
    """ETL pipeline configuration settings"""
    
    # Data Sources
    data_directory: str = Field(default="data/enterprise", description="Directory containing source data files")
    supported_formats: List[str] = Field(default=["csv", "json", "parquet"], description="Supported data formats")
    
    # Processing Settings
    batch_size: int = Field(default=1000, description="Batch size for processing records")
    max_workers: int = Field(default=4, description="Maximum number of worker processes")
    chunk_size: int = Field(default=10000, description="Chunk size for reading large files")
    
    # Data Validation
    validate_data: bool = Field(default=True, description="Enable data validation")
    strict_validation: bool = Field(default=False, description="Enable strict validation mode")
    skip_invalid_records: bool = Field(default=True, description="Skip invalid records instead of failing")
    
    # Performance Settings
    memory_limit_mb: int = Field(default=1024, description="Memory limit for processing in MB")
    temp_directory: str = Field(default="/tmp/etl", description="Temporary directory for processing")
    cleanup_temp_files: bool = Field(default=True, description="Clean up temporary files after processing")
    
    # Data Quality
    min_quality_score: float = Field(default=0.8, description="Minimum quality score for data acceptance")
    max_null_percentage: float = Field(default=0.1, description="Maximum percentage of null values allowed")
    duplicate_threshold: float = Field(default=0.95, description="Similarity threshold for duplicate detection")
    
    # File Processing
    file_patterns: Dict[str, str] = Field(
        default={
            "iot_sensor": "iot_sensor_data.csv",
            "satellite": "satellite_data.csv", 
            "supply_chain": "supply_chain_data.csv",
            "financial": "financial_data.csv",
            "anomaly": "anomaly_data.csv"
        },
        description="File patterns for different data types"
    )
    
    # Error Handling
    max_errors: int = Field(default=100, description="Maximum number of errors before stopping")
    error_log_file: str = Field(default="logs/etl_errors.log", description="Error log file path")
    retry_failed: bool = Field(default=True, description="Retry failed operations")
    max_retries: int = Field(default=3, description="Maximum number of retries")
    
    # Monitoring
    enable_progress_tracking: bool = Field(default=True, description="Enable progress tracking")
    progress_interval: int = Field(default=1000, description="Progress update interval in records")
    enable_metrics: bool = Field(default=True, description="Enable metrics collection")
    
    def get_file_path(self, data_type: str) -> str:
        """Get file path for specific data type"""
        filename = self.file_patterns.get(data_type, f"{data_type}_data.csv")
        return f"{self.data_directory}/{filename}"
    
    def get_temp_file_path(self, data_type: str, suffix: str = "") -> str:
        """Get temporary file path for processing"""
        import os
        os.makedirs(self.temp_directory, exist_ok=True)
        filename = f"{data_type}_temp{suffix}.csv"
        return os.path.join(self.temp_directory, filename)
