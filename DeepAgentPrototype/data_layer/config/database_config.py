"""
Database Configuration

Configuration settings for database connections and management.
"""

import os
from typing import Optional
from pydantic import BaseModel, Field


class DatabaseConfig(BaseModel):
    """Database configuration settings"""
    
    # MongoDB Configuration
    mongodb_host: str = Field(default="localhost", description="MongoDB host")
    mongodb_port: int = Field(default=27017, description="MongoDB port")
    mongodb_database: str = Field(default="organic_agriculture", description="Database name")
    mongodb_username: Optional[str] = Field(default=None, description="MongoDB username")
    mongodb_password: Optional[str] = Field(default=None, description="MongoDB password")
    mongodb_auth_source: str = Field(default="admin", description="MongoDB auth source")
    
    # Connection Pool Settings
    max_pool_size: int = Field(default=100, description="Maximum connection pool size")
    min_pool_size: int = Field(default=10, description="Minimum connection pool size")
    max_idle_time_ms: int = Field(default=30000, description="Max idle time in milliseconds")
    connect_timeout_ms: int = Field(default=20000, description="Connection timeout in milliseconds")
    server_selection_timeout_ms: int = Field(default=5000, description="Server selection timeout")
    
    # Redis Configuration
    redis_host: str = Field(default="localhost", description="Redis host")
    redis_port: int = Field(default=6379, description="Redis port")
    redis_password: Optional[str] = Field(default=None, description="Redis password")
    redis_db: int = Field(default=0, description="Redis database number")
    
    # Performance Settings
    batch_size: int = Field(default=1000, description="Default batch size for operations")
    max_retries: int = Field(default=3, description="Maximum retry attempts")
    retry_delay: float = Field(default=1.0, description="Delay between retries in seconds")
    
    @classmethod
    def from_env(cls) -> "DatabaseConfig":
        """Create configuration from environment variables"""
        return cls(
            mongodb_host=os.getenv("MONGODB_HOST", "localhost"),
            mongodb_port=int(os.getenv("MONGODB_PORT", "27017")),
            mongodb_database=os.getenv("MONGODB_DATABASE", "organic_agriculture"),
            mongodb_username=os.getenv("MONGODB_USERNAME"),
            mongodb_password=os.getenv("MONGODB_PASSWORD"),
            mongodb_auth_source=os.getenv("MONGODB_AUTH_SOURCE", "admin"),
            max_pool_size=int(os.getenv("MONGODB_MAX_POOL_SIZE", "100")),
            min_pool_size=int(os.getenv("MONGODB_MIN_POOL_SIZE", "10")),
            redis_host=os.getenv("REDIS_HOST", "localhost"),
            redis_port=int(os.getenv("REDIS_PORT", "6379")),
            redis_password=os.getenv("REDIS_PASSWORD"),
            redis_db=int(os.getenv("REDIS_DB", "0")),
            batch_size=int(os.getenv("BATCH_SIZE", "1000")),
            max_retries=int(os.getenv("MAX_RETRIES", "3")),
            retry_delay=float(os.getenv("RETRY_DELAY", "1.0"))
        )
    
    def get_mongodb_uri(self) -> str:
        """Get MongoDB connection URI"""
        if self.mongodb_username and self.mongodb_password:
            return f"mongodb://{self.mongodb_username}:{self.mongodb_password}@{self.mongodb_host}:{self.mongodb_port}/{self.mongodb_database}?authSource={self.mongodb_auth_source}"
        else:
            return f"mongodb://{self.mongodb_host}:{self.mongodb_port}/{self.mongodb_database}"
    
    def get_redis_uri(self) -> str:
        """Get Redis connection URI"""
        if self.redis_password:
            return f"redis://:{self.redis_password}@{self.redis_host}:{self.redis_port}/{self.redis_db}"
        else:
            return f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"
