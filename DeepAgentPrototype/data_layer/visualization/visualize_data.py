#!/usr/bin/env python3
"""
Comprehensive Data Visualization Script

Principal AI Engineer level data visualization for the Organic Agriculture system.
"""

import os
import sys
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from datetime import datetime
import logging
from pathlib import Path

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, project_root)

from data_layer.database.mongodb.mongodb_setup import MongoDBManager
from data_layer.visualization.data_explorer import DataExplorer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_visualization_report():
    """Create comprehensive visualization report"""
    logger.info("🚀 Starting Data Visualization Report Generation")
    
    # Initialize MongoDB manager
    mongo_manager = MongoDBManager()
    
    try:
        # Connect to MongoDB
        if not mongo_manager.connect():
            logger.error("❌ Failed to connect to MongoDB")
            return
        
        # Initialize data explorer
        data_explorer = DataExplorer(mongo_manager)
        
        # Load enterprise data
        logger.info("📊 Loading enterprise data...")
        datasets = data_explorer.load_enterprise_data()
        
        if not datasets:
            logger.error("❌ No data loaded")
            return
        
        logger.info(f"✅ Loaded {len(datasets)} datasets")
        
        # Create output directory
        output_dir = Path("visualization_output")
        output_dir.mkdir(exist_ok=True)
        
        # Generate visualizations
        logger.info("🎨 Generating visualizations...")
        
        # 1. Data Overview Dashboard
        logger.info("📊 Creating data overview dashboard...")
        overview_fig = data_explorer.create_data_overview_dashboard(datasets)
        overview_fig.write_html(str(output_dir / "data_overview_dashboard.html"))
        overview_fig.write_image(str(output_dir / "data_overview_dashboard.png"), width=1200, height=800)
        
        # 2. IoT Sensor Analysis
        if 'iot_sensor' in datasets:
            logger.info("📡 Creating IoT sensor analysis...")
            iot_fig = data_explorer.create_iot_sensor_analysis(datasets['iot_sensor'])
            iot_fig.write_html(str(output_dir / "iot_sensor_analysis.html"))
            iot_fig.write_image(str(output_dir / "iot_sensor_analysis.png"), width=1200, height=800)
        
        # 3. Supply Chain Analysis
        if 'supply_chain' in datasets:
            logger.info("🚛 Creating supply chain analysis...")
            sc_fig = data_explorer.create_supply_chain_analysis(datasets['supply_chain'])
            sc_fig.write_html(str(output_dir / "supply_chain_analysis.html"))
            sc_fig.write_image(str(output_dir / "supply_chain_analysis.png"), width=1200, height=800)
        
        # 4. Financial Analysis
        if 'financial' in datasets:
            logger.info("💰 Creating financial analysis...")
            fin_fig = data_explorer.create_financial_analysis(datasets['financial'])
            fin_fig.write_html(str(output_dir / "financial_analysis.html"))
            fin_fig.write_image(str(output_dir / "financial_analysis.png"), width=1200, height=800)
        
        # 5. Anomaly Analysis
        if 'anomaly' in datasets:
            logger.info("⚠️ Creating anomaly analysis...")
            anomaly_fig = data_explorer.create_anomaly_analysis(datasets['anomaly'])
            anomaly_fig.write_html(str(output_dir / "anomaly_analysis.html"))
            anomaly_fig.write_image(str(output_dir / "anomaly_analysis.png"), width=1200, height=800)
        
        # 6. Generate insights report
        logger.info("📋 Generating insights report...")
        insights = data_explorer.generate_data_insights_report(datasets)
        
        # Save insights to JSON
        import json
        with open(output_dir / "data_insights_report.json", 'w') as f:
            json.dump(insights, f, indent=2, default=str)
        
        # Create summary report
        create_summary_report(insights, output_dir)
        
        logger.info("✅ Visualization report generation completed!")
        logger.info(f"📁 Output saved to: {output_dir.absolute()}")
        
        # Print summary
        print_summary(insights)
        
    except Exception as e:
        logger.error(f"❌ Error generating visualization report: {e}")
    
    finally:
        mongo_manager.close_connection()


def create_summary_report(insights, output_dir):
    """Create a summary report in markdown format"""
    report_path = output_dir / "data_summary_report.md"
    
    with open(report_path, 'w') as f:
        f.write("# 🌾 Organic Agriculture AI - Data Summary Report\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Generated by:** Principal AI Engineer\n\n")
        
        # Summary section
        f.write("## 📊 Executive Summary\n\n")
        summary = insights["summary"]
        f.write(f"- **Total Datasets:** {summary['total_datasets']}\n")
        f.write(f"- **Total Records:** {summary['total_records']:,}\n")
        f.write(f"- **Data Types:** {', '.join(summary['data_types'])}\n")
        f.write(f"- **Date Range:** {summary['date_range']['start']} to {summary['date_range']['end']}\n")
        f.write(f"- **Span:** {summary['date_range']['span_days']} days\n\n")
        
        # Quality metrics section
        f.write("## 🎯 Data Quality Metrics\n\n")
        f.write("| Dataset | Completeness | Records | Memory Usage |\n")
        f.write("|---------|-------------|---------|-------------|\n")
        
        for dataset, metrics in insights["quality_metrics"].items():
            f.write(f"| {dataset} | {metrics['completeness']}% | {metrics['total_records']:,} | {metrics['memory_usage']} |\n")
        
        f.write("\n")
        
        # Business insights section
        f.write("## 💼 Business Insights\n\n")
        for category, insight in insights["business_insights"].items():
            f.write(f"### {category.replace('_', ' ').title()}\n")
            for key, value in insight.items():
                f.write(f"- **{key.replace('_', ' ').title()}:** {value}\n")
            f.write("\n")
        
        # Recommendations section
        f.write("## 💡 Recommendations\n\n")
        for i, recommendation in enumerate(insights["recommendations"], 1):
            f.write(f"{i}. {recommendation}\n")
        
        f.write("\n")
        f.write("## 📁 Generated Files\n\n")
        f.write("- `data_overview_dashboard.html` - Interactive overview dashboard\n")
        f.write("- `iot_sensor_analysis.html` - IoT sensor data analysis\n")
        f.write("- `supply_chain_analysis.html` - Supply chain data analysis\n")
        f.write("- `financial_analysis.html` - Financial data analysis\n")
        f.write("- `anomaly_analysis.html` - Anomaly data analysis\n")
        f.write("- `data_insights_report.json` - Detailed insights in JSON format\n")
        f.write("- `data_summary_report.md` - This summary report\n")


def print_summary(insights):
    """Print summary to console"""
    print("\n" + "="*60)
    print("🌾 ORGANIC AGRICULTURE AI - DATA VISUALIZATION SUMMARY")
    print("="*60)
    
    summary = insights["summary"]
    print(f"📊 Total Datasets: {summary['total_datasets']}")
    print(f"📈 Total Records: {summary['total_records']:,}")
    print(f"📅 Date Range: {summary['date_range']['start']} to {summary['date_range']['end']}")
    print(f"⏱️  Span: {summary['date_range']['span_days']} days")
    
    print("\n🎯 Data Quality Metrics:")
    for dataset, metrics in insights["quality_metrics"].items():
        print(f"  {dataset}: {metrics['completeness']}% completeness, {metrics['total_records']:,} records")
    
    print("\n💼 Business Insights:")
    for category, insight in insights["business_insights"].items():
        print(f"  {category}:")
        for key, value in insight.items():
            print(f"    {key}: {value}")
    
    print("\n💡 Key Recommendations:")
    for i, recommendation in enumerate(insights["recommendations"], 1):
        print(f"  {i}. {recommendation}")
    
    print("\n✅ Visualization report generation completed!")
    print("📁 Check the 'visualization_output' directory for all generated files")


def main():
    """Main function"""
    print("🌾 Organic Agriculture AI - Data Visualization")
    print("Principal AI Engineer Level Data Analysis")
    print("="*60)
    
    try:
        create_visualization_report()
    except KeyboardInterrupt:
        print("\n🛑 Visualization interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
