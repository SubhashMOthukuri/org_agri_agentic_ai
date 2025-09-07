"""
Data Validator

Comprehensive data validation utilities for ensuring data quality and consistency.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import pandas as pd
import numpy as np

from ..config.validation_config import ValidationConfig, ValidationRule
from ..models import SensorType, AnomalyType, SeverityLevel, SupplyChainStatus, TransportationMode, MarketType, CropType

logger = logging.getLogger(__name__)


class DataValidator:
    """Data validation utility class"""
    
    def __init__(self, config: ValidationConfig):
        self.config = config
        self.validation_errors = []
        self.validation_stats = {
            "total_records": 0,
            "valid_records": 0,
            "invalid_records": 0,
            "errors_by_type": {},
            "errors_by_field": {}
        }
    
    def validate_record(self, record: Dict[str, Any], data_type: str) -> Tuple[bool, List[str]]:
        """Validate a single record against rules for data type"""
        errors = []
        rules = self.config.get_rules_for_data_type(data_type)
        
        for rule in rules:
            field_value = record.get(rule.field)
            
            # Check if required field is present
            if rule.required and (field_value is None or field_value == ""):
                errors.append(f"Required field '{rule.field}' is missing or empty")
                continue
            
            # Skip validation if field is not present and not required
            if field_value is None or field_value == "":
                continue
            
            # Apply validation rule
            is_valid, error_msg = self._apply_validation_rule(field_value, rule)
            if not is_valid:
                errors.append(f"Field '{rule.field}': {error_msg}")
        
        return len(errors) == 0, errors
    
    def _apply_validation_rule(self, value: Any, rule: ValidationRule) -> Tuple[bool, str]:
        """Apply specific validation rule to value"""
        try:
            if rule.rule_type == "not_empty":
                return value is not None and str(value).strip() != "", rule.error_message
            
            elif rule.rule_type == "numeric_range":
                min_val = rule.parameters.get("min", float('-inf'))
                max_val = rule.parameters.get("max", float('inf'))
                numeric_value = float(value)
                is_valid = min_val <= numeric_value <= max_val
                return is_valid, f"Value {numeric_value} not in range [{min_val}, {max_val}]"
            
            elif rule.rule_type == "enum":
                valid_values = rule.parameters.get("values", [])
                is_valid = str(value).lower() in [v.lower() for v in valid_values]
                return is_valid, f"Value '{value}' not in allowed values: {valid_values}"
            
            elif rule.rule_type == "datetime":
                if isinstance(value, str):
                    pd.to_datetime(value)
                elif isinstance(value, datetime):
                    pass
                else:
                    return False, f"Invalid datetime format: {value}"
                return True, ""
            
            elif rule.rule_type == "email":
                import re
                email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
                is_valid = re.match(email_pattern, str(value)) is not None
                return is_valid, f"Invalid email format: {value}"
            
            elif rule.rule_type == "uuid":
                import uuid
                try:
                    uuid.UUID(str(value))
                    return True, ""
                except ValueError:
                    return False, f"Invalid UUID format: {value}"
            
            else:
                logger.warning(f"Unknown validation rule type: {rule.rule_type}")
                return True, ""
                
        except Exception as e:
            return False, f"Validation error: {str(e)}"
    
    def validate_dataframe(self, df: pd.DataFrame, data_type: str) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
        """Validate entire DataFrame and return valid records with error details"""
        valid_records = []
        invalid_records = []
        
        for idx, row in df.iterrows():
            record = row.to_dict()
            is_valid, errors = self.validate_record(record, data_type)
            
            if is_valid:
                valid_records.append(record)
            else:
                invalid_record = {
                    "index": idx,
                    "record": record,
                    "errors": errors
                }
                invalid_records.append(invalid_record)
                
                # Update error statistics
                self.validation_stats["errors_by_type"][data_type] = self.validation_stats["errors_by_type"].get(data_type, 0) + 1
                for error in errors:
                    field = error.split("'")[1] if "'" in error else "unknown"
                    self.validation_stats["errors_by_field"][field] = self.validation_stats["errors_by_field"].get(field, 0) + 1
        
        # Update overall statistics
        self.validation_stats["total_records"] += len(df)
        self.validation_stats["valid_records"] += len(valid_records)
        self.validation_stats["invalid_records"] += len(invalid_records)
        
        valid_df = pd.DataFrame(valid_records) if valid_records else pd.DataFrame()
        
        return valid_df, invalid_records
    
    def validate_cross_dataset_consistency(self, datasets: Dict[str, pd.DataFrame]) -> List[str]:
        """Validate consistency across multiple datasets"""
        errors = []
        
        if not self.config.enable_cross_validation:
            return errors
        
        # Check foreign key relationships
        if self.config.foreign_key_validation:
            errors.extend(self._validate_foreign_keys(datasets))
        
        # Check temporal consistency
        if self.config.temporal_consistency:
            errors.extend(self._validate_temporal_consistency(datasets))
        
        return errors
    
    def _validate_foreign_keys(self, datasets: Dict[str, pd.DataFrame]) -> List[str]:
        """Validate foreign key relationships between datasets"""
        errors = []
        
        # Check farm_id consistency
        farm_ids = set()
        for data_type, df in datasets.items():
            if 'farm_id' in df.columns:
                current_farm_ids = set(df['farm_id'].dropna().unique())
                if farm_ids and not farm_ids.issubset(current_farm_ids):
                    missing_ids = farm_ids - current_farm_ids
                    errors.append(f"Missing farm_ids in {data_type}: {missing_ids}")
                farm_ids.update(current_farm_ids)
        
        return errors
    
    def _validate_temporal_consistency(self, datasets: Dict[str, pd.DataFrame]) -> List[str]:
        """Validate temporal consistency across datasets"""
        errors = []
        
        timestamps = {}
        for data_type, df in datasets.items():
            if 'timestamp' in df.columns:
                timestamps[data_type] = pd.to_datetime(df['timestamp']).dropna()
        
        if len(timestamps) > 1:
            # Check for reasonable temporal overlap
            min_times = {dt: ts.min() for dt, ts in timestamps.items()}
            max_times = {dt: ts.max() for dt, ts in timestamps.items()}
            
            overall_min = max(min_times.values())
            overall_max = min(max_times.values())
            
            if overall_min > overall_max:
                errors.append(f"Temporal inconsistency: datasets don't overlap in time")
        
        return errors
    
    def get_validation_report(self) -> Dict[str, Any]:
        """Get comprehensive validation report"""
        total = self.validation_stats["total_records"]
        valid = self.validation_stats["valid_records"]
        invalid = self.validation_stats["invalid_records"]
        
        report = {
            "summary": {
                "total_records": total,
                "valid_records": valid,
                "invalid_records": invalid,
                "validation_rate": valid / total if total > 0 else 0,
                "error_rate": invalid / total if total > 0 else 0
            },
            "errors_by_type": self.validation_stats["errors_by_type"],
            "errors_by_field": self.validation_stats["errors_by_field"],
            "config": {
                "validation_enabled": self.config.enable_validation,
                "strict_mode": self.config.strict_mode,
                "cross_validation": self.config.enable_cross_validation
            }
        }
        
        return report
    
    def reset_stats(self):
        """Reset validation statistics"""
        self.validation_stats = {
            "total_records": 0,
            "valid_records": 0,
            "invalid_records": 0,
            "errors_by_type": {},
            "errors_by_field": {}
        }
        self.validation_errors = []
