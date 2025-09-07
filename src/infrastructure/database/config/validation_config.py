"""
Data Validation Configuration

Configuration settings for data validation and quality assurance.
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class ValidationRule(BaseModel):
    """Individual validation rule"""
    field: str = Field(..., description="Field name to validate")
    rule_type: str = Field(..., description="Type of validation rule")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Rule parameters")
    required: bool = Field(default=True, description="Whether field is required")
    error_message: str = Field(default="Validation failed", description="Error message for failed validation")


class ValidationConfig(BaseModel):
    """Data validation configuration settings"""
    
    # General Settings
    enable_validation: bool = Field(default=True, description="Enable data validation")
    strict_mode: bool = Field(default=False, description="Enable strict validation mode")
    validate_on_insert: bool = Field(default=True, description="Validate data on insert")
    validate_on_update: bool = Field(default=True, description="Validate data on update")
    
    # Quality Thresholds
    min_completeness: float = Field(default=0.95, description="Minimum data completeness threshold")
    max_outlier_percentage: float = Field(default=0.05, description="Maximum outlier percentage")
    min_consistency_score: float = Field(default=0.90, description="Minimum consistency score")
    
    # Field Validation Rules
    iot_sensor_rules: List[ValidationRule] = Field(
        default_factory=lambda: [
            ValidationRule(field="sensor_id", rule_type="not_empty", required=True),
            ValidationRule(field="farm_id", rule_type="not_empty", required=True),
            ValidationRule(field="value", rule_type="numeric_range", parameters={"min": -100, "max": 1000}),
            ValidationRule(field="quality_score", rule_type="numeric_range", parameters={"min": 0, "max": 1}),
            ValidationRule(field="timestamp", rule_type="datetime", required=True)
        ]
    )
    
    satellite_rules: List[ValidationRule] = Field(
        default_factory=lambda: [
            ValidationRule(field="farm_id", rule_type="not_empty", required=True),
            ValidationRule(field="vegetation_index", rule_type="numeric_range", parameters={"min": 0, "max": 1}),
            ValidationRule(field="cloud_cover", rule_type="numeric_range", parameters={"min": 0, "max": 1}),
            ValidationRule(field="quality_score", rule_type="numeric_range", parameters={"min": 0, "max": 1})
        ]
    )
    
    supply_chain_rules: List[ValidationRule] = Field(
        default_factory=lambda: [
            ValidationRule(field="supply_chain_id", rule_type="not_empty", required=True),
            ValidationRule(field="farm_id", rule_type="not_empty", required=True),
            ValidationRule(field="status", rule_type="enum", parameters={"values": ["pending", "in_transit", "delivered", "delayed", "cancelled"]}),
            ValidationRule(field="risk_score", rule_type="numeric_range", parameters={"min": 0, "max": 1})
        ]
    )
    
    financial_rules: List[ValidationRule] = Field(
        default_factory=lambda: [
            ValidationRule(field="farm_id", rule_type="not_empty", required=True),
            ValidationRule(field="crop_type", rule_type="not_empty", required=True),
            ValidationRule(field="market_price", rule_type="numeric_range", parameters={"min": 0, "max": 10000}),
            ValidationRule(field="profit_margin", rule_type="numeric_range", parameters={"min": -1, "max": 1})
        ]
    )
    
    anomaly_rules: List[ValidationRule] = Field(
        default_factory=lambda: [
            ValidationRule(field="anomaly_id", rule_type="not_empty", required=True),
            ValidationRule(field="farm_id", rule_type="not_empty", required=True),
            ValidationRule(field="anomaly_type", rule_type="not_empty", required=True),
            ValidationRule(field="severity", rule_type="enum", parameters={"values": ["low", "medium", "high", "critical"]}),
            ValidationRule(field="confidence_score", rule_type="numeric_range", parameters={"min": 0, "max": 1})
        ]
    )
    
    # Cross-Dataset Validation
    enable_cross_validation: bool = Field(default=True, description="Enable cross-dataset validation")
    foreign_key_validation: bool = Field(default=True, description="Validate foreign key relationships")
    temporal_consistency: bool = Field(default=True, description="Validate temporal consistency")
    
    # Error Handling
    max_validation_errors: int = Field(default=1000, description="Maximum validation errors before stopping")
    error_reporting: str = Field(default="detailed", description="Error reporting level: basic, detailed, verbose")
    log_validation_errors: bool = Field(default=True, description="Log validation errors")
    validation_log_file: str = Field(default="logs/validation_errors.log", description="Validation error log file")
    
    def get_rules_for_data_type(self, data_type: str) -> List[ValidationRule]:
        """Get validation rules for specific data type"""
        rules_map = {
            "iot_sensor": self.iot_sensor_rules,
            "satellite": self.satellite_rules,
            "supply_chain": self.supply_chain_rules,
            "financial": self.financial_rules,
            "anomaly": self.anomaly_rules
        }
        return rules_map.get(data_type, [])
    
    def is_field_required(self, data_type: str, field_name: str) -> bool:
        """Check if field is required for specific data type"""
        rules = self.get_rules_for_data_type(data_type)
        for rule in rules:
            if rule.field == field_name:
                return rule.required
        return False
