"""
Backend Configuration - Organic Agriculture Agentic AI

Configuration settings for the FastAPI backend including database,
authentication, CORS, and other service configurations.

Author: Principal AI Engineer
Version: 1.0.0
Date: December 2024
"""

import os
from typing import List, Optional
from pydantic import BaseSettings, Field


class Settings(BaseSettings):
    """Application settings"""
    
    # Application settings
    app_name: str = "Organic Agriculture Agentic AI"
    app_version: str = "1.0.0"
    debug: bool = Field(default=False, env="DEBUG")
    
    # Server settings
    host: str = Field(default="0.0.0.0", env="API_HOST")
    port: int = Field(default=8000, env="API_PORT")
    debug: bool = Field(default=True, env="API_DEBUG")
    
    # Database settings
    mongodb_uri: str = Field(default="mongodb://localhost:27017/", env="MONGODB_URI")
    mongodb_database: str = Field(default="organic_agriculture_ai", env="MONGODB_DATABASE")
    
    # Redis settings (for caching)
    redis_url: str = Field(default="redis://localhost:6379/0", env="REDIS_URL")
    
    # Authentication settings
    secret_key: str = Field(default="your_secret_key_here", env="SECRET_KEY")
    jwt_secret: str = Field(default="your_jwt_secret_here", env="JWT_SECRET")
    algorithm: str = Field(default="HS256", env="JWT_ALGORITHM")
    jwt_expiration_hours: int = Field(default=24, env="JWT_EXPIRATION_HOURS")
    
    # CORS settings
    cors_origins: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:3001"],
        env="CORS_ORIGINS"
    )
    cors_allow_credentials: bool = Field(default=True, env="CORS_ALLOW_CREDENTIALS")
    cors_allow_methods: List[str] = Field(default=["*"], env="CORS_ALLOW_METHODS")
    cors_allow_headers: List[str] = Field(default=["*"], env="CORS_ALLOW_HEADERS")
    
    # Trusted hosts
    trusted_hosts: List[str] = Field(
        default=["localhost", "127.0.0.1", "*.yourdomain.com"],
        env="TRUSTED_HOSTS"
    )
    
    # Logging settings
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_file: str = Field(default="logs/agriculture_ai.log", env="LOG_FILE")
    
    # Agent settings
    agent_timeout: int = Field(default=30, env="AGENT_TIMEOUT")
    max_concurrent_agents: int = Field(default=10, env="MAX_CONCURRENT_AGENTS")
    
    # WebSocket settings
    websocket_ping_interval: int = Field(default=25, env="WEBSOCKET_PING_INTERVAL")
    websocket_ping_timeout: int = Field(default=20, env="WEBSOCKET_PING_TIMEOUT")
    max_websocket_connections: int = Field(default=100, env="MAX_WEBSOCKET_CONNECTIONS")
    
    # Rate limiting
    rate_limit_requests: int = Field(default=100, env="RATE_LIMIT_REQUESTS")
    rate_limit_window: int = Field(default=60, env="RATE_LIMIT_WINDOW")  # seconds
    
    # Metrics settings
    metrics_enabled: bool = Field(default=True, env="METRICS_ENABLED")
    metrics_collection_interval: int = Field(default=60, env="METRICS_COLLECTION_INTERVAL")
    
    # External API settings
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    
    # File upload settings
    max_file_size: int = Field(default=10 * 1024 * 1024, env="MAX_FILE_SIZE")  # 10MB
    allowed_file_types: List[str] = Field(
        default=["image/jpeg", "image/png", "application/pdf", "text/csv"],
        env="ALLOWED_FILE_TYPES"
    )
    
    # Cache settings
    cache_ttl: int = Field(default=300, env="CACHE_TTL")  # 5 minutes
    cache_max_size: int = Field(default=1000, env="CACHE_MAX_SIZE")
    
    # Monitoring settings
    health_check_interval: int = Field(default=30, env="HEALTH_CHECK_INTERVAL")
    performance_monitoring: bool = Field(default=True, env="PERFORMANCE_MONITORING")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get application settings"""
    return settings


def get_database_url() -> str:
    """Get MongoDB connection URL"""
    return settings.mongodb_uri


def get_redis_url() -> str:
    """Get Redis connection URL"""
    return settings.redis_url


# Environment-specific configurations
class DevelopmentSettings(Settings):
    """Development environment settings"""
    debug: bool = True
    reload: bool = True
    log_level: str = "DEBUG"
    cors_origins: List[str] = ["http://localhost:3000", "http://localhost:3001", "http://127.0.0.1:3000"]


class ProductionSettings(Settings):
    """Production environment settings"""
    debug: bool = False
    reload: bool = False
    log_level: str = "WARNING"
    cors_origins: List[str] = ["https://yourdomain.com", "https://www.yourdomain.com"]


class TestingSettings(Settings):
    """Testing environment settings"""
    debug: bool = True
    mongodb_database: str = "organic_agriculture_test"
    redis_db: int = 1
    log_level: str = "DEBUG"


def get_environment_settings(env: str = "development") -> Settings:
    """Get environment-specific settings"""
    if env == "production":
        return ProductionSettings()
    elif env == "testing":
        return TestingSettings()
    else:
        return DevelopmentSettings()


# Configuration validation
def validate_config():
    """Validate configuration settings"""
    errors = []
    
    # Check required settings
    if not settings.secret_key or settings.secret_key == "your-secret-key-change-in-production":
        errors.append("SECRET_KEY must be set to a secure value in production")
    
    if settings.debug and settings.secret_key == "your-secret-key-change-in-production":
        errors.append("SECRET_KEY should be changed from default value")
    
    # Check database settings
    if not settings.mongodb_host:
        errors.append("MONGODB_HOST must be set")
    
    if not settings.mongodb_database:
        errors.append("MONGODB_DATABASE must be set")
    
    # Check CORS settings
    if not settings.cors_origins:
        errors.append("CORS_ORIGINS must be set")
    
    if errors:
        raise ValueError(f"Configuration validation failed: {'; '.join(errors)}")
    
    return True


if __name__ == "__main__":
    # Test configuration
    try:
        validate_config()
        print("✅ Configuration is valid")
        print(f"App: {settings.app_name} v{settings.app_version}")
        print(f"Database: {get_database_url()}")
        print(f"Redis: {get_redis_url()}")
        print(f"Debug: {settings.debug}")
    except Exception as e:
        print(f"❌ Configuration error: {e}")
