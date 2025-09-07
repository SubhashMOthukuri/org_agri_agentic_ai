"""
Enterprise Data Analysis and Quality Assessment
Analyzes the generated enterprise datasets for production readiness
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class EnterpriseDataAnalyzer:
    """Analyzes enterprise-level datasets for quality and production readiness"""
    
    def __init__(self, data_dir: str = "data/enterprise"):
        self.data_dir = Path(data_dir)
        self.datasets = {}
        self.analysis_results = {}
        
    def load_datasets(self):
        """Load all enterprise datasets"""
        print("📊 Loading enterprise datasets...")
        
        dataset_files = {
            "iot_sensor": "iot_sensor_data.csv",
            "satellite": "satellite_data.csv", 
            "supply_chain": "supply_chain_data.csv",
            "financial": "financial_data.csv",
            "anomalies": "anomaly_data.csv"
        }
        
        for name, filename in dataset_files.items():
            file_path = self.data_dir / filename
            if file_path.exists():
                print(f"   Loading {name}...")
                self.datasets[name] = pd.read_csv(file_path)
                print(f"   ✅ {name}: {len(self.datasets[name]):,} records")
            else:
                print(f"   ❌ {name}: File not found")
        
        print(f"📈 Total datasets loaded: {len(self.datasets)}")
        return self.datasets
    
    def analyze_data_quality(self):
        """Analyze data quality across all datasets"""
        print("\n🔍 Analyzing data quality...")
        
        quality_metrics = {}
        
        for name, df in self.datasets.items():
            print(f"\n📊 Analyzing {name} dataset...")
            
            # Basic statistics
            total_records = len(df)
            total_columns = len(df.columns)
            
            # Completeness analysis
            missing_data = df.isnull().sum()
            completeness = (1 - missing_data.sum() / (total_records * total_columns)) * 100
            
            # Data type consistency
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            categorical_columns = df.select_dtypes(include=['object']).columns
            
            # Outlier detection (for numeric columns)
            outlier_percentage = 0
            if len(numeric_columns) > 0:
                outlier_counts = 0
                for col in numeric_columns:
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
                    outlier_counts += len(outliers)
                outlier_percentage = (outlier_counts / (total_records * len(numeric_columns))) * 100
            
            # Duplicate records
            duplicates = df.duplicated().sum()
            duplicate_percentage = (duplicates / total_records) * 100
            
            # Memory usage
            memory_usage = df.memory_usage(deep=True).sum() / 1024 / 1024  # MB
            
            quality_metrics[name] = {
                "total_records": total_records,
                "total_columns": total_columns,
                "completeness": round(completeness, 2),
                "missing_data_percentage": round((100 - completeness), 2),
                "numeric_columns": len(numeric_columns),
                "categorical_columns": len(categorical_columns),
                "outlier_percentage": round(outlier_percentage, 2),
                "duplicate_percentage": round(duplicate_percentage, 2),
                "memory_usage_mb": round(memory_usage, 2),
                "data_quality_score": round(self._calculate_quality_score(
                    completeness, outlier_percentage, duplicate_percentage
                ), 2)
            }
            
            print(f"   ✅ Completeness: {completeness:.1f}%")
            print(f"   ✅ Data Quality Score: {quality_metrics[name]['data_quality_score']:.1f}/100")
            print(f"   ✅ Memory Usage: {memory_usage:.1f} MB")
        
        self.analysis_results["quality_metrics"] = quality_metrics
        return quality_metrics
    
    def analyze_iot_sensor_data(self):
        """Detailed analysis of IoT sensor data"""
        if "iot_sensor" not in self.datasets:
            return None
        
        print("\n🔌 Analyzing IoT sensor data...")
        df = self.datasets["iot_sensor"]
        
        # Sensor type distribution
        sensor_distribution = df["sensor_type"].value_counts()
        
        # Sensor status analysis
        status_distribution = df["sensor_status"].value_counts()
        failure_rate = (status_distribution.get("failed", 0) / len(df)) * 100
        
        # Battery level analysis
        battery_stats = df["battery_level"].describe()
        
        # Signal strength analysis
        signal_stats = df["signal_strength"].describe()
        
        # Farm coverage
        farms_covered = df["farm_id"].nunique()
        
        # Time coverage
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        time_span = (df["timestamp"].max() - df["timestamp"].min()).days
        
        iot_analysis = {
            "total_readings": len(df),
            "sensor_types": len(sensor_distribution),
            "farms_covered": farms_covered,
            "time_span_days": time_span,
            "sensor_distribution": sensor_distribution.to_dict(),
            "status_distribution": status_distribution.to_dict(),
            "failure_rate": round(failure_rate, 2),
            "battery_stats": battery_stats.to_dict(),
            "signal_stats": signal_stats.to_dict(),
            "readings_per_farm": round(len(df) / farms_covered, 0),
            "readings_per_day": round(len(df) / time_span, 0)
        }
        
        print(f"   ✅ Total readings: {len(df):,}")
        print(f"   ✅ Sensor types: {len(sensor_distribution)}")
        print(f"   ✅ Farms covered: {farms_covered}")
        print(f"   ✅ Failure rate: {failure_rate:.1f}%")
        print(f"   ✅ Time span: {time_span} days")
        
        self.analysis_results["iot_analysis"] = iot_analysis
        return iot_analysis
    
    def analyze_satellite_data(self):
        """Detailed analysis of satellite data"""
        if "satellite" not in self.datasets:
            return None
        
        print("\n🛰️ Analyzing satellite data...")
        df = self.datasets["satellite"]
        
        # Vegetation indices analysis
        vegetation_columns = ["ndvi", "ndwi", "evi", "savi", "gci"]
        vegetation_stats = {}
        
        for col in vegetation_columns:
            if col in df.columns:
                vegetation_stats[col] = {
                    "mean": round(df[col].mean(), 4),
                    "std": round(df[col].std(), 4),
                    "min": round(df[col].min(), 4),
                    "max": round(df[col].max(), 4)
                }
        
        # Crop health analysis
        if "crop_health_score" in df.columns:
            health_stats = df["crop_health_score"].describe()
            health_distribution = pd.cut(df["crop_health_score"], 
                                       bins=[0, 30, 60, 80, 100], 
                                       labels=["Poor", "Fair", "Good", "Excellent"]).value_counts()
        else:
            health_stats = None
            health_distribution = None
        
        # Growth stage distribution
        if "growth_stage" in df.columns:
            growth_distribution = df["growth_stage"].value_counts()
        else:
            growth_distribution = None
        
        # Cloud cover analysis
        if "cloud_cover" in df.columns:
            cloud_stats = df["cloud_cover"].describe()
            clear_images = (df["cloud_cover"] < 0.1).sum()
            clear_percentage = (clear_images / len(df)) * 100
        else:
            cloud_stats = None
            clear_percentage = None
        
        satellite_analysis = {
            "total_observations": len(df),
            "farms_covered": df["farm_id"].nunique(),
            "vegetation_indices": vegetation_stats,
            "health_stats": health_stats.to_dict() if health_stats is not None else None,
            "health_distribution": health_distribution.to_dict() if health_distribution is not None else None,
            "growth_distribution": growth_distribution.to_dict() if growth_distribution is not None else None,
            "cloud_stats": cloud_stats.to_dict() if cloud_stats is not None else None,
            "clear_images_percentage": round(clear_percentage, 2) if clear_percentage is not None else None
        }
        
        print(f"   ✅ Total observations: {len(df):,}")
        print(f"   ✅ Farms covered: {df['farm_id'].nunique()}")
        if clear_percentage is not None:
            print(f"   ✅ Clear images: {clear_percentage:.1f}%")
        
        self.analysis_results["satellite_analysis"] = satellite_analysis
        return satellite_analysis
    
    def analyze_supply_chain_data(self):
        """Detailed analysis of supply chain data"""
        if "supply_chain" not in self.datasets:
            return None
        
        print("\n🚛 Analyzing supply chain data...")
        df = self.datasets["supply_chain"]
        
        # Event type distribution
        event_distribution = df["event_type"].value_counts()
        
        # Crop distribution
        crop_distribution = df["crop"].value_counts()
        
        # Delivery status analysis
        status_distribution = df["delivery_status"].value_counts()
        on_time_delivery = (status_distribution.get("delivered", 0) / len(df)) * 100
        
        # Cost analysis
        cost_columns = ["price_per_ton", "transport_cost", "storage_cost", "fuel_cost"]
        cost_stats = {}
        for col in cost_columns:
            if col in df.columns:
                cost_stats[col] = {
                    "mean": round(df[col].mean(), 2),
                    "median": round(df[col].median(), 2),
                    "std": round(df[col].std(), 2)
                }
        
        # Risk factors analysis
        if "risk_factors" in df.columns:
            risk_distribution = df["risk_factors"].value_counts()
        else:
            risk_distribution = None
        
        # Quality grade analysis
        if "quality_grade" in df.columns:
            quality_distribution = df["quality_grade"].value_counts()
        else:
            quality_distribution = None
        
        supply_chain_analysis = {
            "total_events": len(df),
            "event_types": len(event_distribution),
            "crops_tracked": len(crop_distribution),
            "event_distribution": event_distribution.to_dict(),
            "crop_distribution": crop_distribution.to_dict(),
            "status_distribution": status_distribution.to_dict(),
            "on_time_delivery_rate": round(on_time_delivery, 2),
            "cost_analysis": cost_stats,
            "risk_distribution": risk_distribution.to_dict() if risk_distribution is not None else None,
            "quality_distribution": quality_distribution.to_dict() if quality_distribution is not None else None
        }
        
        print(f"   ✅ Total events: {len(df):,}")
        print(f"   ✅ Event types: {len(event_distribution)}")
        print(f"   ✅ Crops tracked: {len(crop_distribution)}")
        print(f"   ✅ On-time delivery: {on_time_delivery:.1f}%")
        
        self.analysis_results["supply_chain_analysis"] = supply_chain_analysis
        return supply_chain_analysis
    
    def analyze_financial_data(self):
        """Detailed analysis of financial data"""
        if "financial" not in self.datasets:
            return None
        
        print("\n💰 Analyzing financial data...")
        df = self.datasets["financial"]
        
        # Revenue analysis
        revenue_stats = df["revenue"].describe()
        
        # Profit analysis
        if "gross_profit" in df.columns:
            profit_stats = df["gross_profit"].describe()
            profitable_farms = (df["gross_profit"] > 0).sum()
            profitability_rate = (profitable_farms / len(df)) * 100
        else:
            profit_stats = None
            profitability_rate = None
        
        # Cost structure analysis
        cost_columns = [col for col in df.columns if "cost" in col.lower()]
        cost_analysis = {}
        for col in cost_columns:
            cost_analysis[col] = {
                "mean": round(df[col].mean(), 2),
                "median": round(df[col].median(), 2),
                "std": round(df[col].std(), 2)
            }
        
        # Market analysis
        market_columns = [col for col in df.columns if "market" in col.lower() or "demand" in col.lower()]
        market_analysis = {}
        for col in market_columns:
            market_analysis[col] = {
                "mean": round(df[col].mean(), 3),
                "std": round(df[col].std(), 3)
            }
        
        financial_analysis = {
            "total_records": len(df),
            "farms_analyzed": df["farm_id"].nunique(),
            "revenue_stats": revenue_stats.to_dict(),
            "profit_stats": profit_stats.to_dict() if profit_stats is not None else None,
            "profitability_rate": round(profitability_rate, 2) if profitability_rate is not None else None,
            "cost_analysis": cost_analysis,
            "market_analysis": market_analysis
        }
        
        print(f"   ✅ Total records: {len(df):,}")
        print(f"   ✅ Farms analyzed: {df['farm_id'].nunique()}")
        if profitability_rate is not None:
            print(f"   ✅ Profitability rate: {profitability_rate:.1f}%")
        
        self.analysis_results["financial_analysis"] = financial_analysis
        return financial_analysis
    
    def analyze_anomaly_data(self):
        """Detailed analysis of anomaly data"""
        if "anomalies" not in self.datasets:
            return None
        
        print("\n⚠️ Analyzing anomaly data...")
        df = self.datasets["anomalies"]
        
        # Anomaly type distribution
        anomaly_distribution = df["anomaly_type"].value_counts()
        
        # Severity analysis
        if "severity" in df.columns:
            severity_distribution = df["severity"].value_counts()
        else:
            severity_distribution = None
        
        # Affected systems analysis
        if "affected_systems" in df.columns:
            # Count unique systems affected
            all_systems = []
            for systems in df["affected_systems"]:
                if isinstance(systems, str):
                    all_systems.extend(systems.split(","))
            system_counts = pd.Series(all_systems).value_counts()
        else:
            system_counts = None
        
        # Loss analysis
        if "estimated_loss" in df.columns:
            loss_stats = df["estimated_loss"].describe()
            total_loss = df["estimated_loss"].sum()
        else:
            loss_stats = None
            total_loss = None
        
        # Recovery time analysis
        if "recovery_time" in df.columns:
            recovery_stats = df["recovery_time"].describe()
        else:
            recovery_stats = None
        
        anomaly_analysis = {
            "total_anomalies": len(df),
            "anomaly_types": len(anomaly_distribution),
            "anomaly_distribution": anomaly_distribution.to_dict(),
            "severity_distribution": severity_distribution.to_dict() if severity_distribution is not None else None,
            "affected_systems": system_counts.to_dict() if system_counts is not None else None,
            "loss_stats": loss_stats.to_dict() if loss_stats is not None else None,
            "total_estimated_loss": round(total_loss, 2) if total_loss is not None else None,
            "recovery_stats": recovery_stats.to_dict() if recovery_stats is not None else None
        }
        
        print(f"   ✅ Total anomalies: {len(df):,}")
        print(f"   ✅ Anomaly types: {len(anomaly_distribution)}")
        if total_loss is not None:
            print(f"   ✅ Total estimated loss: ${total_loss:,.2f}")
        
        self.analysis_results["anomaly_analysis"] = anomaly_analysis
        return anomaly_analysis
    
    def _calculate_quality_score(self, completeness, outlier_percentage, duplicate_percentage):
        """Calculate overall data quality score"""
        # Weighted scoring: completeness (50%), outliers (30%), duplicates (20%)
        completeness_score = completeness
        outlier_score = max(0, 100 - outlier_percentage * 2)  # Penalize outliers heavily
        duplicate_score = max(0, 100 - duplicate_percentage * 5)  # Penalize duplicates heavily
        
        quality_score = (completeness_score * 0.5 + 
                        outlier_score * 0.3 + 
                        duplicate_score * 0.2)
        
        return min(100, max(0, quality_score))
    
    def generate_enterprise_report(self):
        """Generate comprehensive enterprise data analysis report"""
        print("\n📊 Generating enterprise analysis report...")
        
        # Load and analyze all datasets
        self.load_datasets()
        self.analyze_data_quality()
        self.analyze_iot_sensor_data()
        self.analyze_satellite_data()
        self.analyze_supply_chain_data()
        self.analyze_financial_data()
        self.analyze_anomaly_data()
        
        # Calculate overall enterprise readiness score
        quality_scores = [metrics["data_quality_score"] for metrics in self.analysis_results["quality_metrics"].values()]
        overall_quality_score = np.mean(quality_scores)
        
        # Generate enterprise readiness assessment
        enterprise_readiness = self._assess_enterprise_readiness()
        
        # Create comprehensive report
        report = {
            "report_timestamp": datetime.now().isoformat(),
            "enterprise_readiness_score": round(overall_quality_score, 2),
            "enterprise_readiness_assessment": enterprise_readiness,
            "total_datasets": len(self.datasets),
            "total_records": sum(len(df) for df in self.datasets.values()),
            "analysis_results": self.analysis_results,
            "recommendations": self._generate_recommendations()
        }
        
        # Save report
        report_path = self.data_dir / "enterprise_analysis_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"✅ Enterprise analysis report saved: {report_path}")
        
        # Display summary
        self._display_enterprise_summary(report)
        
        return report
    
    def _assess_enterprise_readiness(self):
        """Assess enterprise production readiness"""
        readiness_criteria = {
            "data_volume": self._assess_data_volume(),
            "data_quality": self._assess_data_quality(),
            "data_diversity": self._assess_data_diversity(),
            "real_time_capability": self._assess_real_time_capability(),
            "scalability": self._assess_scalability(),
            "monitoring": self._assess_monitoring()
        }
        
        overall_score = np.mean(list(readiness_criteria.values()))
        
        return {
            "overall_score": round(overall_score, 2),
            "criteria": readiness_criteria,
            "readiness_level": self._get_readiness_level(overall_score),
            "production_ready": overall_score >= 80
        }
    
    def _assess_data_volume(self):
        """Assess data volume adequacy"""
        total_records = sum(len(df) for df in self.datasets.values())
        if total_records >= 1000000:  # 1M+ records
            return 100
        elif total_records >= 100000:  # 100K+ records
            return 80
        elif total_records >= 10000:  # 10K+ records
            return 60
        else:
            return 40
    
    def _assess_data_quality(self):
        """Assess data quality"""
        if "quality_metrics" not in self.analysis_results:
            return 50
        
        quality_scores = [metrics["data_quality_score"] for metrics in self.analysis_results["quality_metrics"].values()]
        return np.mean(quality_scores)
    
    def _assess_data_diversity(self):
        """Assess data diversity"""
        diversity_score = 0
        
        # Check for different data types
        if "iot_sensor" in self.datasets:
            diversity_score += 20
        if "satellite" in self.datasets:
            diversity_score += 20
        if "supply_chain" in self.datasets:
            diversity_score += 20
        if "financial" in self.datasets:
            diversity_score += 20
        if "anomalies" in self.datasets:
            diversity_score += 20
        
        return diversity_score
    
    def _assess_real_time_capability(self):
        """Assess real-time data capability"""
        # Check for time-series data and sensor data
        if "iot_sensor" in self.datasets:
            return 90  # IoT data indicates real-time capability
        return 50
    
    def _assess_scalability(self):
        """Assess system scalability"""
        total_records = sum(len(df) for df in self.datasets.values())
        if total_records >= 1000000:
            return 100
        elif total_records >= 100000:
            return 80
        else:
            return 60
    
    def _assess_monitoring(self):
        """Assess monitoring and alerting capability"""
        if "anomalies" in self.datasets:
            return 90  # Anomaly data indicates monitoring capability
        return 50
    
    def _get_readiness_level(self, score):
        """Get readiness level based on score"""
        if score >= 90:
            return "Production Ready"
        elif score >= 80:
            return "Near Production Ready"
        elif score >= 70:
            return "Development Ready"
        elif score >= 60:
            return "Prototype Ready"
        else:
            return "Not Ready"
    
    def _generate_recommendations(self):
        """Generate recommendations for enterprise deployment"""
        recommendations = []
        
        # Data volume recommendations
        total_records = sum(len(df) for df in self.datasets.values())
        if total_records < 1000000:
            recommendations.append("Scale data generation to 1M+ records for production ML training")
        
        # Real-time recommendations
        if "iot_sensor" not in self.datasets:
            recommendations.append("Implement real-time IoT data streaming with Kafka")
        
        # Monitoring recommendations
        if "anomalies" not in self.datasets:
            recommendations.append("Add comprehensive monitoring and alerting systems")
        
        # Infrastructure recommendations
        recommendations.extend([
            "Set up MongoDB cluster for data storage",
            "Implement Redis for real-time caching",
            "Deploy Apache Kafka for data streaming",
            "Set up monitoring with Prometheus and Grafana",
            "Implement data backup and disaster recovery",
            "Add data encryption and security measures"
        ])
        
        return recommendations
    
    def _display_enterprise_summary(self, report):
        """Display enterprise analysis summary"""
        print("\n" + "=" * 80)
        print("🏢 ENTERPRISE DATA ANALYSIS REPORT")
        print("=" * 80)
        
        print(f"📊 Overall Readiness Score: {report['enterprise_readiness_score']}/100")
        print(f"🎯 Readiness Level: {report['enterprise_readiness_assessment']['readiness_level']}")
        print(f"🚀 Production Ready: {'✅ YES' if report['enterprise_readiness_assessment']['production_ready'] else '❌ NO'}")
        print()
        
        print(f"📈 Dataset Summary:")
        print(f"   - Total Datasets: {report['total_datasets']}")
        print(f"   - Total Records: {report['total_records']:,}")
        print()
        
        print("📊 Quality Assessment:")
        for dataset, metrics in report['analysis_results']['quality_metrics'].items():
            print(f"   - {dataset}: {metrics['data_quality_score']}/100")
        print()
        
        print("🎯 Key Recommendations:")
        for i, rec in enumerate(report['recommendations'][:5], 1):
            print(f"   {i}. {rec}")
        print()

def main():
    """Run enterprise data analysis"""
    analyzer = EnterpriseDataAnalyzer()
    report = analyzer.generate_enterprise_report()
    return report

if __name__ == "__main__":
    main()
