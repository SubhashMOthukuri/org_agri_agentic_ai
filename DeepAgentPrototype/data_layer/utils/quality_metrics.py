"""
Data Quality Metrics

Utilities for calculating and monitoring data quality metrics.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class QualityMetrics:
    """Data quality metrics calculator"""
    
    def __init__(self):
        self.metrics_history = []
    
    def calculate_completeness(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate data completeness metrics"""
        total_cells = df.size
        null_cells = df.isnull().sum().sum()
        completeness = 1 - (null_cells / total_cells) if total_cells > 0 else 0
        
        # Per-column completeness
        column_completeness = {}
        for column in df.columns:
            null_count = df[column].isnull().sum()
            total_count = len(df)
            column_completeness[column] = 1 - (null_count / total_count) if total_count > 0 else 0
        
        return {
            "overall_completeness": completeness,
            "column_completeness": column_completeness,
            "total_cells": total_cells,
            "null_cells": null_cells
        }
    
    def calculate_consistency(self, df: pd.DataFrame, data_type: str) -> Dict[str, Any]:
        """Calculate data consistency metrics"""
        consistency_metrics = {}
        
        # Data type consistency
        type_consistency = self._check_data_type_consistency(df)
        consistency_metrics["type_consistency"] = type_consistency
        
        # Value range consistency
        range_consistency = self._check_value_range_consistency(df, data_type)
        consistency_metrics["range_consistency"] = range_consistency
        
        # Format consistency
        format_consistency = self._check_format_consistency(df)
        consistency_metrics["format_consistency"] = format_consistency
        
        # Overall consistency score
        overall_score = np.mean([
            type_consistency["score"],
            range_consistency["score"],
            format_consistency["score"]
        ])
        consistency_metrics["overall_consistency"] = overall_score
        
        return consistency_metrics
    
    def calculate_accuracy(self, df: pd.DataFrame, data_type: str) -> Dict[str, Any]:
        """Calculate data accuracy metrics"""
        accuracy_metrics = {}
        
        # Outlier detection
        outliers = self._detect_outliers(df, data_type)
        accuracy_metrics["outlier_percentage"] = outliers["percentage"]
        accuracy_metrics["outlier_count"] = outliers["count"]
        
        # Duplicate detection
        duplicates = self._detect_duplicates(df)
        accuracy_metrics["duplicate_percentage"] = duplicates["percentage"]
        accuracy_metrics["duplicate_count"] = duplicates["count"]
        
        # Temporal accuracy (for time-series data)
        if 'timestamp' in df.columns:
            temporal_accuracy = self._check_temporal_accuracy(df)
            accuracy_metrics["temporal_accuracy"] = temporal_accuracy
        
        # Overall accuracy score
        accuracy_score = 1 - (outliers["percentage"] + duplicates["percentage"]) / 2
        accuracy_metrics["overall_accuracy"] = max(0, accuracy_score)
        
        return accuracy_metrics
    
    def calculate_timeliness(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate data timeliness metrics"""
        timeliness_metrics = {}
        
        if 'timestamp' not in df.columns:
            return {"error": "No timestamp column found"}
        
        # Convert timestamps
        timestamps = pd.to_datetime(df['timestamp'], errors='coerce')
        valid_timestamps = timestamps.dropna()
        
        if len(valid_timestamps) == 0:
            return {"error": "No valid timestamps found"}
        
        current_time = datetime.utcnow()
        time_diffs = (current_time - valid_timestamps).dt.total_seconds() / 3600  # hours
        
        # Freshness metrics
        timeliness_metrics["oldest_record_hours"] = time_diffs.max()
        timeliness_metrics["newest_record_hours"] = time_diffs.min()
        timeliness_metrics["average_age_hours"] = time_diffs.mean()
        
        # Freshness score (higher is better, 1.0 for very recent data)
        freshness_score = max(0, 1 - (time_diffs.mean() / 24))  # 24 hours = 0 score
        timeliness_metrics["freshness_score"] = min(1.0, freshness_score)
        
        # Data frequency analysis
        if len(valid_timestamps) > 1:
            time_diffs_sorted = np.diff(valid_timestamps.sort_values().dt.total_seconds() / 3600)
            timeliness_metrics["average_interval_hours"] = np.mean(time_diffs_sorted)
            timeliness_metrics["interval_consistency"] = 1 - (np.std(time_diffs_sorted) / np.mean(time_diffs_sorted)) if np.mean(time_diffs_sorted) > 0 else 0
        
        return timeliness_metrics
    
    def calculate_uniqueness(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate data uniqueness metrics"""
        total_records = len(df)
        unique_records = len(df.drop_duplicates())
        
        uniqueness_metrics = {
            "total_records": total_records,
            "unique_records": unique_records,
            "duplicate_records": total_records - unique_records,
            "uniqueness_ratio": unique_records / total_records if total_records > 0 else 0
        }
        
        # Column-level uniqueness
        column_uniqueness = {}
        for column in df.columns:
            unique_values = df[column].nunique()
            total_values = len(df[column].dropna())
            column_uniqueness[column] = unique_values / total_values if total_values > 0 else 0
        
        uniqueness_metrics["column_uniqueness"] = column_uniqueness
        
        return uniqueness_metrics
    
    def calculate_overall_quality_score(self, df: pd.DataFrame, data_type: str) -> Dict[str, Any]:
        """Calculate overall data quality score"""
        # Calculate individual metrics
        completeness = self.calculate_completeness(df)
        consistency = self.calculate_consistency(df, data_type)
        accuracy = self.calculate_accuracy(df, data_type)
        timeliness = self.calculate_timeliness(df)
        uniqueness = self.calculate_uniqueness(df)
        
        # Weighted overall score
        weights = {
            "completeness": 0.25,
            "consistency": 0.20,
            "accuracy": 0.25,
            "timeliness": 0.15,
            "uniqueness": 0.15
        }
        
        overall_score = (
            completeness["overall_completeness"] * weights["completeness"] +
            consistency["overall_consistency"] * weights["consistency"] +
            accuracy["overall_accuracy"] * weights["accuracy"] +
            timeliness.get("freshness_score", 0.5) * weights["timeliness"] +
            uniqueness["uniqueness_ratio"] * weights["uniqueness"]
        )
        
        return {
            "overall_score": overall_score,
            "completeness": completeness,
            "consistency": consistency,
            "accuracy": accuracy,
            "timeliness": timeliness,
            "uniqueness": uniqueness,
            "weights": weights
        }
    
    def _check_data_type_consistency(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check data type consistency across columns"""
        type_issues = []
        
        for column in df.columns:
            # Check for mixed types
            if df[column].dtype == 'object':
                # Try to convert to numeric
                try:
                    pd.to_numeric(df[column], errors='raise')
                except:
                    # Check if it's datetime
                    try:
                        pd.to_datetime(df[column], errors='raise')
                    except:
                        type_issues.append(f"Column '{column}' has inconsistent data types")
        
        score = 1 - (len(type_issues) / len(df.columns)) if len(df.columns) > 0 else 1
        
        return {
            "score": score,
            "issues": type_issues,
            "total_columns": len(df.columns)
        }
    
    def _check_value_range_consistency(self, df: pd.DataFrame, data_type: str) -> Dict[str, Any]:
        """Check value range consistency"""
        range_issues = []
        
        # Define expected ranges for different data types
        range_rules = {
            "iot_sensor": {
                "value": (-100, 1000),
                "quality_score": (0, 1),
                "battery_level": (0, 100)
            },
            "satellite": {
                "vegetation_index": (0, 1),
                "cloud_cover": (0, 1),
                "quality_score": (0, 1)
            },
            "supply_chain": {
                "risk_score": (0, 1)
            },
            "financial": {
                "market_price": (0, 10000),
                "profit_margin": (-1, 1)
            },
            "anomaly": {
                "confidence_score": (0, 1)
            }
        }
        
        rules = range_rules.get(data_type, {})
        
        for column, (min_val, max_val) in rules.items():
            if column in df.columns:
                numeric_values = pd.to_numeric(df[column], errors='coerce').dropna()
                if len(numeric_values) > 0:
                    out_of_range = ((numeric_values < min_val) | (numeric_values > max_val)).sum()
                    if out_of_range > 0:
                        range_issues.append(f"Column '{column}': {out_of_range} values out of range [{min_val}, {max_val}]")
        
        score = 1 - (len(range_issues) / len(rules)) if len(rules) > 0 else 1
        
        return {
            "score": score,
            "issues": range_issues,
            "rules_checked": len(rules)
        }
    
    def _check_format_consistency(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check format consistency"""
        format_issues = []
        
        for column in df.columns:
            if df[column].dtype == 'object':
                # Check for consistent string formats
                unique_formats = df[column].str.len().nunique()
                if unique_formats > 1:
                    format_issues.append(f"Column '{column}' has inconsistent string lengths")
        
        score = 1 - (len(format_issues) / len(df.columns)) if len(df.columns) > 0 else 1
        
        return {
            "score": score,
            "issues": format_issues,
            "total_columns": len(df.columns)
        }
    
    def _detect_outliers(self, df: pd.DataFrame, data_type: str) -> Dict[str, Any]:
        """Detect outliers in numeric columns"""
        outlier_count = 0
        total_numeric_values = 0
        
        for column in df.select_dtypes(include=[np.number]).columns:
            values = df[column].dropna()
            if len(values) > 0:
                Q1 = values.quantile(0.25)
                Q3 = values.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = ((values < lower_bound) | (values > upper_bound)).sum()
                outlier_count += outliers
                total_numeric_values += len(values)
        
        percentage = (outlier_count / total_numeric_values) if total_numeric_values > 0 else 0
        
        return {
            "count": outlier_count,
            "percentage": percentage,
            "total_values": total_numeric_values
        }
    
    def _detect_duplicates(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect duplicate records"""
        total_records = len(df)
        duplicate_count = total_records - len(df.drop_duplicates())
        percentage = duplicate_count / total_records if total_records > 0 else 0
        
        return {
            "count": duplicate_count,
            "percentage": percentage,
            "total_records": total_records
        }
    
    def _check_temporal_accuracy(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check temporal accuracy for time-series data"""
        timestamps = pd.to_datetime(df['timestamp'], errors='coerce').dropna()
        
        if len(timestamps) < 2:
            return {"score": 1.0, "issues": []}
        
        # Check for reasonable time intervals
        time_diffs = timestamps.sort_values().diff().dt.total_seconds() / 3600  # hours
        time_diffs = time_diffs.dropna()
        
        # Flag extremely short or long intervals as potential issues
        issues = []
        if len(time_diffs) > 0:
            median_interval = time_diffs.median()
            extreme_short = (time_diffs < median_interval * 0.1).sum()
            extreme_long = (time_diffs > median_interval * 10).sum()
            
            if extreme_short > 0:
                issues.append(f"{extreme_short} extremely short intervals detected")
            if extreme_long > 0:
                issues.append(f"{extreme_long} extremely long intervals detected")
        
        score = 1 - (len(issues) / 2) if issues else 1  # Penalize for temporal issues
        
        return {
            "score": score,
            "issues": issues,
            "median_interval_hours": time_diffs.median() if len(time_diffs) > 0 else 0
        }
