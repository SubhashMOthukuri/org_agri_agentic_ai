"""
Metrics Configuration - Centralized configuration for all project metrics

This module contains configuration settings for metrics collection, thresholds,
and monitoring across the Organic Agriculture Agentic AI project.

Author: Principal AI Engineer
Version: 1.0.0
Date: December 2024
"""

from dataclasses import dataclass
from typing import Dict, List, Any
from enum import Enum


class AlertLevel(Enum):
    """Alert levels for metrics monitoring"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class MetricConfig:
    """Configuration for individual metrics"""
    name: str
    enabled: bool = True
    collection_interval: int = 60  # seconds
    retention_days: int = 30
    alert_enabled: bool = True
    alert_level: AlertLevel = AlertLevel.WARNING
    description: str = ""


class MetricsConfig:
    """Centralized metrics configuration"""
    
    def __init__(self):
        self.metric_configs = self._initialize_metric_configs()
        self.collection_settings = self._initialize_collection_settings()
        self.alert_settings = self._initialize_alert_settings()
        self.export_settings = self._initialize_export_settings()
    
    def _initialize_metric_configs(self) -> Dict[str, MetricConfig]:
        """Initialize metric configurations"""
        return {
            # Data Quality Metrics
            "data_completeness": MetricConfig(
                name="data_completeness",
                collection_interval=300,  # 5 minutes
                retention_days=90,
                alert_level=AlertLevel.CRITICAL,
                description="Data completeness percentage"
            ),
            "data_consistency": MetricConfig(
                name="data_consistency",
                collection_interval=300,
                retention_days=90,
                alert_level=AlertLevel.CRITICAL,
                description="Data consistency across datasets"
            ),
            "data_accuracy": MetricConfig(
                name="data_accuracy",
                collection_interval=300,
                retention_days=90,
                alert_level=AlertLevel.WARNING,
                description="Data accuracy based on validation rules"
            ),
            "duplicate_rate": MetricConfig(
                name="duplicate_rate",
                collection_interval=600,  # 10 minutes
                retention_days=30,
                alert_level=AlertLevel.WARNING,
                description="Percentage of duplicate records"
            ),
            "outlier_rate": MetricConfig(
                name="outlier_rate",
                collection_interval=600,
                retention_days=30,
                alert_level=AlertLevel.INFO,
                description="Percentage of outlier records"
            ),
            
            # Performance Metrics
            "data_generation_speed": MetricConfig(
                name="data_generation_speed",
                collection_interval=60,
                retention_days=7,
                alert_level=AlertLevel.WARNING,
                description="Data generation processing speed"
            ),
            "memory_usage": MetricConfig(
                name="memory_usage",
                collection_interval=30,
                retention_days=7,
                alert_level=AlertLevel.CRITICAL,
                description="Peak memory usage during processing"
            ),
            "processing_time": MetricConfig(
                name="processing_time",
                collection_interval=60,
                retention_days=7,
                alert_level=AlertLevel.WARNING,
                description="Total processing time for data generation"
            ),
            "database_query_time": MetricConfig(
                name="database_query_time",
                collection_interval=30,
                retention_days=14,
                alert_level=AlertLevel.WARNING,
                description="Average database query execution time"
            ),
            "api_response_time": MetricConfig(
                name="api_response_time",
                collection_interval=30,
                retention_days=14,
                alert_level=AlertLevel.WARNING,
                description="Average API response time"
            ),
            
            # Business Metrics
            "enterprise_readiness_score": MetricConfig(
                name="enterprise_readiness_score",
                collection_interval=3600,  # 1 hour
                retention_days=365,
                alert_level=AlertLevel.CRITICAL,
                description="Overall enterprise readiness score"
            ),
            "data_volume": MetricConfig(
                name="data_volume",
                collection_interval=3600,
                retention_days=365,
                alert_level=AlertLevel.INFO,
                description="Total number of data records"
            ),
            "farms_covered": MetricConfig(
                name="farms_covered",
                collection_interval=3600,
                retention_days=90,
                alert_level=AlertLevel.INFO,
                description="Number of farms covered by the system"
            ),
            "crop_types": MetricConfig(
                name="crop_types",
                collection_interval=3600,
                retention_days=90,
                alert_level=AlertLevel.INFO,
                description="Number of different crop types supported"
            ),
            "sensor_types": MetricConfig(
                name="sensor_types",
                collection_interval=3600,
                retention_days=90,
                alert_level=AlertLevel.INFO,
                description="Number of different sensor types"
            ),
            
            # System Metrics
            "uptime": MetricConfig(
                name="uptime",
                collection_interval=60,
                retention_days=90,
                alert_level=AlertLevel.CRITICAL,
                description="System uptime percentage"
            ),
            "error_rate": MetricConfig(
                name="error_rate",
                collection_interval=60,
                retention_days=30,
                alert_level=AlertLevel.CRITICAL,
                description="System error rate"
            ),
            "cpu_usage": MetricConfig(
                name="cpu_usage",
                collection_interval=30,
                retention_days=14,
                alert_level=AlertLevel.WARNING,
                description="CPU usage percentage"
            ),
            "disk_usage": MetricConfig(
                name="disk_usage",
                collection_interval=60,
                retention_days=30,
                alert_level=AlertLevel.WARNING,
                description="Disk usage percentage"
            ),
            "network_latency": MetricConfig(
                name="network_latency",
                collection_interval=30,
                retention_days=14,
                alert_level=AlertLevel.WARNING,
                description="Network latency"
            ),
            
            # ML Model Metrics
            "model_accuracy": MetricConfig(
                name="model_accuracy",
                collection_interval=1800,  # 30 minutes
                retention_days=90,
                alert_level=AlertLevel.CRITICAL,
                description="ML model accuracy"
            ),
            "model_precision": MetricConfig(
                name="model_precision",
                collection_interval=1800,
                retention_days=90,
                alert_level=AlertLevel.WARNING,
                description="ML model precision"
            ),
            "model_recall": MetricConfig(
                name="model_recall",
                collection_interval=1800,
                retention_days=90,
                alert_level=AlertLevel.WARNING,
                description="ML model recall"
            ),
            "model_f1_score": MetricConfig(
                name="model_f1_score",
                collection_interval=1800,
                retention_days=90,
                alert_level=AlertLevel.WARNING,
                description="ML model F1 score"
            ),
            "prediction_latency": MetricConfig(
                name="prediction_latency",
                collection_interval=300,
                retention_days=30,
                alert_level=AlertLevel.WARNING,
                description="ML model prediction latency"
            ),
            
            # Agent Metrics
            "agent_response_time": MetricConfig(
                name="agent_response_time",
                collection_interval=60,
                retention_days=30,
                alert_level=AlertLevel.WARNING,
                description="Average agent response time"
            ),
            "agent_success_rate": MetricConfig(
                name="agent_success_rate",
                collection_interval=300,
                retention_days=30,
                alert_level=AlertLevel.CRITICAL,
                description="Agent task success rate"
            ),
            "agent_availability": MetricConfig(
                name="agent_availability",
                collection_interval=60,
                retention_days=30,
                alert_level=AlertLevel.CRITICAL,
                description="Agent availability percentage"
            ),
            "decision_accuracy": MetricConfig(
                name="decision_accuracy",
                collection_interval=1800,
                retention_days=60,
                alert_level=AlertLevel.WARNING,
                description="Decision agent accuracy"
            ),
            "alert_response_time": MetricConfig(
                name="alert_response_time",
                collection_interval=60,
                retention_days=30,
                alert_level=AlertLevel.WARNING,
                description="Alert response time"
            ),
            
            # Infrastructure Metrics
            "database_connections": MetricConfig(
                name="database_connections",
                collection_interval=30,
                retention_days=14,
                alert_level=AlertLevel.WARNING,
                description="Database connection pool usage"
            ),
            "cache_hit_rate": MetricConfig(
                name="cache_hit_rate",
                collection_interval=60,
                retention_days=14,
                alert_level=AlertLevel.WARNING,
                description="Cache hit rate"
            ),
            "storage_usage": MetricConfig(
                name="storage_usage",
                collection_interval=300,
                retention_days=30,
                alert_level=AlertLevel.WARNING,
                description="Storage usage percentage"
            ),
            "backup_success_rate": MetricConfig(
                name="backup_success_rate",
                collection_interval=3600,
                retention_days=90,
                alert_level=AlertLevel.CRITICAL,
                description="Backup success rate"
            ),
            "replication_lag": MetricConfig(
                name="replication_lag",
                collection_interval=60,
                retention_days=14,
                alert_level=AlertLevel.WARNING,
                description="Database replication lag"
            )
        }
    
    def _initialize_collection_settings(self) -> Dict[str, Any]:
        """Initialize data collection settings"""
        return {
            "default_interval": 60,  # seconds
            "batch_size": 1000,
            "max_retries": 3,
            "timeout": 30,  # seconds
            "parallel_collection": True,
            "max_workers": 4,
            "collection_timeout": 300,  # 5 minutes
            "enable_historical_data": True,
            "historical_data_days": 7
        }
    
    def _initialize_alert_settings(self) -> Dict[str, Any]:
        """Initialize alert settings"""
        return {
            "enabled": True,
            "email_alerts": True,
            "slack_alerts": False,
            "webhook_alerts": False,
            "alert_cooldown": 300,  # 5 minutes
            "max_alerts_per_hour": 10,
            "alert_recipients": ["admin@orgagri.com"],
            "slack_webhook_url": "",
            "webhook_url": "",
            "alert_templates": {
                "critical": "🚨 CRITICAL: {metric_name} is {value} {unit}",
                "warning": "⚠️ WARNING: {metric_name} is {value} {unit}",
                "info": "ℹ️ INFO: {metric_name} is {value} {unit}"
            }
        }
    
    def _initialize_export_settings(self) -> Dict[str, Any]:
        """Initialize export settings"""
        return {
            "formats": ["json", "csv", "excel"],
            "export_directory": "metrics/exports",
            "auto_export": True,
            "export_interval": 3600,  # 1 hour
            "retention_days": 90,
            "compression": True,
            "include_metadata": True,
            "export_schedule": "0 * * * *"  # Every hour
        }
    
    def get_metric_config(self, metric_name: str) -> MetricConfig:
        """Get configuration for a specific metric"""
        return self.metric_configs.get(metric_name, MetricConfig(name=metric_name))
    
    def get_enabled_metrics(self) -> List[str]:
        """Get list of enabled metrics"""
        return [name for name, config in self.metric_configs.items() if config.enabled]
    
    def get_metrics_by_alert_level(self, alert_level: AlertLevel) -> List[str]:
        """Get metrics by alert level"""
        return [
            name for name, config in self.metric_configs.items() 
            if config.alert_level == alert_level and config.enabled
        ]
    
    def get_collection_schedule(self) -> Dict[str, int]:
        """Get collection schedule for all metrics"""
        return {
            name: config.collection_interval 
            for name, config in self.metric_configs.items() 
            if config.enabled
        }
    
    def update_metric_config(self, metric_name: str, **kwargs):
        """Update configuration for a specific metric"""
        if metric_name in self.metric_configs:
            config = self.metric_configs[metric_name]
            for key, value in kwargs.items():
                if hasattr(config, key):
                    setattr(config, key, value)
    
    def disable_metric(self, metric_name: str):
        """Disable a specific metric"""
        self.update_metric_config(metric_name, enabled=False)
    
    def enable_metric(self, metric_name: str):
        """Enable a specific metric"""
        self.update_metric_config(metric_name, enabled=True)
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get configuration summary"""
        enabled_count = len(self.get_enabled_metrics())
        total_count = len(self.metric_configs)
        
        alert_levels = {}
        for level in AlertLevel:
            alert_levels[level.value] = len(self.get_metrics_by_alert_level(level))
        
        return {
            "total_metrics": total_count,
            "enabled_metrics": enabled_count,
            "disabled_metrics": total_count - enabled_count,
            "alert_levels": alert_levels,
            "collection_settings": self.collection_settings,
            "alert_settings": self.alert_settings,
            "export_settings": self.export_settings
        }


# Global configuration instance
metrics_config = MetricsConfig()


def get_metrics_config() -> MetricsConfig:
    """Get the global metrics configuration instance"""
    return metrics_config


if __name__ == "__main__":
    # Display configuration summary
    config = get_metrics_config()
    summary = config.get_config_summary()
    
    print("=" * 80)
    print("🌾 METRICS CONFIGURATION SUMMARY")
    print("=" * 80)
    print(f"Total Metrics: {summary['total_metrics']}")
    print(f"Enabled Metrics: {summary['enabled_metrics']}")
    print(f"Disabled Metrics: {summary['disabled_metrics']}")
    print()
    
    print("📊 METRICS BY ALERT LEVEL")
    print("-" * 40)
    for level, count in summary['alert_levels'].items():
        print(f"{level.upper()}: {count} metrics")
    print()
    
    print("⚙️ COLLECTION SETTINGS")
    print("-" * 40)
    for key, value in summary['collection_settings'].items():
        print(f"{key}: {value}")
    print()
    
    print("🚨 ALERT SETTINGS")
    print("-" * 40)
    for key, value in summary['alert_settings'].items():
        if key != "alert_templates":
            print(f"{key}: {value}")
    print()
    
    print("📁 EXPORT SETTINGS")
    print("-" * 40)
    for key, value in summary['export_settings'].items():
        print(f"{key}: {value}")
    print()
    
    print("✅ Configuration loaded successfully!")
