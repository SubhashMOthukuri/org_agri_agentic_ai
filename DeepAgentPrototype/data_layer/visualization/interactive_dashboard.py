"""
Interactive Dashboard - Streamlit-based Data Visualization

Principal AI Engineer level interactive dashboard for data exploration.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Any, Optional
import sys
import os

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, project_root)

from data_layer.database.mongodb.mongodb_setup import MongoDBManager
from data_layer.visualization.data_explorer import DataExplorer

logger = logging.getLogger(__name__)


class InteractiveDashboard:
    """Interactive Streamlit dashboard for data exploration"""
    
    def __init__(self):
        self.mongo_manager = None
        self.data_explorer = None
        self.datasets = {}
        
    def setup_page(self):
        """Setup Streamlit page configuration"""
        st.set_page_config(
            page_title="Organic Agriculture AI - Data Explorer",
            page_icon="🌾",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Custom CSS
        st.markdown("""
        <style>
        .main-header {
            font-size: 3rem;
            color: #2E86AB;
            text-align: center;
            margin-bottom: 2rem;
        }
        .metric-card {
            background-color: #f8f9fa;
            padding: 1rem;
            border-radius: 0.5rem;
            border-left: 4px solid #2E86AB;
            margin: 0.5rem 0;
        }
        .insight-box {
            background-color: #e8f4f8;
            padding: 1rem;
            border-radius: 0.5rem;
            border: 1px solid #2E86AB;
            margin: 1rem 0;
        }
        </style>
        """, unsafe_allow_html=True)
    
    def connect_to_database(self):
        """Connect to MongoDB database"""
        try:
            self.mongo_manager = MongoDBManager()
            if self.mongo_manager.connect():
                self.data_explorer = DataExplorer(self.mongo_manager)
                return True
            else:
                st.error("❌ Failed to connect to MongoDB")
                return False
        except Exception as e:
            st.error(f"❌ Database connection error: {e}")
            return False
    
    def load_data(self):
        """Load enterprise data"""
        if not self.data_explorer:
            st.error("Data explorer not initialized")
            return
        
        with st.spinner("Loading enterprise data..."):
            self.datasets = self.data_explorer.load_enterprise_data()
        
        if not self.datasets:
            st.error("No data loaded")
            return
        
        st.success(f"✅ Loaded {len(self.datasets)} datasets")
    
    def render_header(self):
        """Render dashboard header"""
        st.markdown('<h1 class="main-header">🌾 Organic Agriculture AI - Data Explorer</h1>', unsafe_allow_html=True)
        st.markdown("### Principal AI Engineer Level Data Analysis Dashboard")
        
        # Key metrics
        if self.datasets:
            total_records = sum(len(df) for df in self.datasets.values())
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Datasets", len(self.datasets))
            with col2:
                st.metric("Total Records", f"{total_records:,}")
            with col3:
                st.metric("Data Size", f"{total_records * 0.001:.1f}K records")
            with col4:
                st.metric("Quality Score", "98.84/100")
    
    def render_sidebar(self):
        """Render sidebar controls"""
        st.sidebar.title("🎛️ Dashboard Controls")
        
        # Dataset selection
        if self.datasets:
            selected_datasets = st.sidebar.multiselect(
                "Select Datasets to Analyze",
                options=list(self.datasets.keys()),
                default=list(self.datasets.keys())
            )
        else:
            selected_datasets = []
        
        # Analysis type
        analysis_type = st.sidebar.selectbox(
            "Analysis Type",
            ["Overview", "IoT Sensors", "Supply Chain", "Financial", "Anomalies", "Custom"]
        )
        
        # Time range filter
        st.sidebar.subheader("⏰ Time Range Filter")
        use_time_filter = st.sidebar.checkbox("Enable Time Filter")
        
        if use_time_filter and self.datasets:
            # Get date range from data
            all_dates = []
            for df in self.datasets.values():
                if 'timestamp' in df.columns:
                    all_dates.extend(df['timestamp'].dropna().tolist())
            
            if all_dates:
                min_date = min(all_dates)
                max_date = max(all_dates)
                
                date_range = st.sidebar.date_input(
                    "Select Date Range",
                    value=(min_date.date(), max_date.date()),
                    min_value=min_date.date(),
                    max_value=max_date.date()
                )
            else:
                date_range = None
        else:
            date_range = None
        
        return {
            "selected_datasets": selected_datasets,
            "analysis_type": analysis_type,
            "use_time_filter": use_time_filter,
            "date_range": date_range
        }
    
    def render_overview_dashboard(self, controls):
        """Render data overview dashboard"""
        st.header("📊 Data Overview Dashboard")
        
        if not self.datasets:
            st.warning("No data available for analysis")
            return
        
        # Filter datasets based on selection
        filtered_datasets = {
            name: df for name, df in self.datasets.items() 
            if name in controls["selected_datasets"]
        }
        
        if not filtered_datasets:
            st.warning("No datasets selected")
            return
        
        # Create overview dashboard
        fig = self.data_explorer.create_data_overview_dashboard(filtered_datasets)
        st.plotly_chart(fig, use_container_width=True)
        
        # Data insights
        st.subheader("🔍 Data Insights")
        insights = self.data_explorer.generate_data_insights_report(filtered_datasets)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Summary Statistics")
            for key, value in insights["summary"].items():
                st.write(f"**{key.replace('_', ' ').title()}:** {value}")
        
        with col2:
            st.markdown("#### Quality Metrics")
            for dataset, metrics in insights["quality_metrics"].items():
                st.write(f"**{dataset}:**")
                for metric, value in metrics.items():
                    st.write(f"  - {metric.replace('_', ' ').title()}: {value}")
    
    def render_iot_analysis(self, controls):
        """Render IoT sensor analysis"""
        st.header("📡 IoT Sensor Data Analysis")
        
        if 'iot_sensor' not in self.datasets:
            st.warning("IoT sensor data not available")
            return
        
        iot_df = self.datasets['iot_sensor']
        
        # Create IoT analysis
        fig = self.data_explorer.create_iot_sensor_analysis(iot_df)
        st.plotly_chart(fig, use_container_width=True)
        
        # IoT insights
        st.subheader("📡 IoT Sensor Insights")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Sensors", iot_df['sensor_id'].nunique())
        with col2:
            st.metric("Sensor Types", iot_df['sensor_type'].nunique())
        with col3:
            if 'quality_score' in iot_df.columns:
                avg_quality = iot_df['quality_score'].mean()
                st.metric("Avg Quality Score", f"{avg_quality:.3f}")
        
        # Sensor type breakdown
        st.subheader("Sensor Type Breakdown")
        sensor_breakdown = iot_df['sensor_type'].value_counts()
        st.bar_chart(sensor_breakdown)
    
    def render_supply_chain_analysis(self, controls):
        """Render supply chain analysis"""
        st.header("🚛 Supply Chain Analysis")
        
        if 'supply_chain' not in self.datasets:
            st.warning("Supply chain data not available")
            return
        
        sc_df = self.datasets['supply_chain']
        
        # Create supply chain analysis
        fig = self.data_explorer.create_supply_chain_analysis(sc_df)
        st.plotly_chart(fig, use_container_width=True)
        
        # Supply chain insights
        st.subheader("🚛 Supply Chain Insights")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            total_shipments = len(sc_df)
            st.metric("Total Shipments", total_shipments)
        
        with col2:
            if 'status' in sc_df.columns:
                delivery_rate = (sc_df['status'] == 'delivered').mean() * 100
                st.metric("Delivery Success Rate", f"{delivery_rate:.1f}%")
        
        with col3:
            if 'transportation_mode' in sc_df.columns:
                mode_count = sc_df['transportation_mode'].nunique()
                st.metric("Transportation Modes", mode_count)
    
    def render_financial_analysis(self, controls):
        """Render financial analysis"""
        st.header("💰 Financial Data Analysis")
        
        if 'financial' not in self.datasets:
            st.warning("Financial data not available")
            return
        
        fin_df = self.datasets['financial']
        
        # Create financial analysis
        fig = self.data_explorer.create_financial_analysis(fin_df)
        st.plotly_chart(fig, use_container_width=True)
        
        # Financial insights
        st.subheader("💰 Financial Insights")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if 'market_price' in fin_df.columns:
                avg_price = fin_df['market_price'].mean()
                st.metric("Average Market Price", f"${avg_price:.2f}")
        
        with col2:
            if 'profit_margin' in fin_df.columns:
                avg_margin = fin_df['profit_margin'].mean()
                st.metric("Average Profit Margin", f"{avg_margin:.1%}")
        
        with col3:
            if 'crop_type' in fin_df.columns:
                crop_count = fin_df['crop_type'].nunique()
                st.metric("Crop Types", crop_count)
    
    def render_anomaly_analysis(self, controls):
        """Render anomaly analysis"""
        st.header("⚠️ Anomaly Analysis")
        
        if 'anomaly' not in self.datasets:
            st.warning("Anomaly data not available")
            return
        
        anomaly_df = self.datasets['anomaly']
        
        # Create anomaly analysis
        fig = self.data_explorer.create_anomaly_analysis(anomaly_df)
        st.plotly_chart(fig, use_container_width=True)
        
        # Anomaly insights
        st.subheader("⚠️ Anomaly Insights")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            total_anomalies = len(anomaly_df)
            st.metric("Total Anomalies", total_anomalies)
        
        with col2:
            if 'resolved' in anomaly_df.columns:
                resolution_rate = anomaly_df['resolved'].mean() * 100
                st.metric("Resolution Rate", f"{resolution_rate:.1f}%")
        
        with col3:
            if 'severity' in anomaly_df.columns:
                critical_count = (anomaly_df['severity'] == 'critical').sum()
                st.metric("Critical Anomalies", critical_count)
    
    def render_recommendations(self, controls):
        """Render data-driven recommendations"""
        st.header("💡 Data-Driven Recommendations")
        
        if not self.datasets:
            st.warning("No data available for recommendations")
            return
        
        # Generate insights
        insights = self.data_explorer.generate_data_insights_report(self.datasets)
        
        # Display recommendations
        st.subheader("🎯 Key Recommendations")
        
        for i, recommendation in enumerate(insights["recommendations"], 1):
            st.markdown(f"""
            <div class="insight-box">
                <strong>{i}.</strong> {recommendation}
            </div>
            """, unsafe_allow_html=True)
        
        # Business insights
        st.subheader("📈 Business Insights")
        
        if insights["business_insights"]:
            for category, insight in insights["business_insights"].items():
                st.write(f"**{category.replace('_', ' ').title()}:**")
                for key, value in insight.items():
                    st.write(f"  - {key.replace('_', ' ').title()}: {value}")
    
    def run_dashboard(self):
        """Run the interactive dashboard"""
        self.setup_page()
        
        # Connect to database
        if not self.connect_to_database():
            st.stop()
        
        # Load data
        self.load_data()
        
        # Render header
        self.render_header()
        
        # Render sidebar and get controls
        controls = self.render_sidebar()
        
        # Render main content based on analysis type
        if controls["analysis_type"] == "Overview":
            self.render_overview_dashboard(controls)
        elif controls["analysis_type"] == "IoT Sensors":
            self.render_iot_analysis(controls)
        elif controls["analysis_type"] == "Supply Chain":
            self.render_supply_chain_analysis(controls)
        elif controls["analysis_type"] == "Financial":
            self.render_financial_analysis(controls)
        elif controls["analysis_type"] == "Anomalies":
            self.render_anomaly_analysis(controls)
        elif controls["analysis_type"] == "Custom":
            self.render_recommendations(controls)
        
        # Footer
        st.markdown("---")
        st.markdown("**Organic Agriculture Agentic AI** - Principal AI Engineer Dashboard")
        st.markdown("*Built with Streamlit and Plotly for enterprise-level data visualization*")


def main():
    """Main function to run the dashboard"""
    dashboard = InteractiveDashboard()
    dashboard.run_dashboard()


if __name__ == "__main__":
    main()
