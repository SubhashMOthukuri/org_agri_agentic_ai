"""
Data Explorer - Comprehensive Data Visualization and Analysis

Principal AI Engineer level data exploration tools for the Organic Agriculture system.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class DataExplorer:
    """Advanced data exploration and visualization tools"""
    
    def __init__(self, mongo_manager=None):
        self.mongo_manager = mongo_manager
        self.colors = {
            'primary': '#2E86AB',
            'secondary': '#A23B72', 
            'success': '#F18F01',
            'warning': '#C73E1D',
            'info': '#6C757D',
            'light': '#F8F9FA',
            'dark': '#343A40'
        }
    
    def load_enterprise_data(self) -> Dict[str, pd.DataFrame]:
        """Load all enterprise datasets from MongoDB"""
        try:
            datasets = {}
            
            if not self.mongo_manager or not hasattr(self.mongo_manager, 'db'):
                logger.error("MongoDB not connected")
                return datasets
            
            # Load IoT Sensor Data
            iot_data = list(self.mongo_manager.db.iot_sensor_data.find().limit(100000))  # Sample for performance
            if iot_data:
                datasets['iot_sensor'] = pd.DataFrame(iot_data)
                datasets['iot_sensor']['timestamp'] = pd.to_datetime(datasets['iot_sensor']['timestamp'])
            
            # Load Satellite Data
            satellite_data = list(self.mongo_manager.db.satellite_data.find())
            if satellite_data:
                datasets['satellite'] = pd.DataFrame(satellite_data)
                datasets['satellite']['timestamp'] = pd.to_datetime(datasets['satellite']['timestamp'])
            
            # Load Supply Chain Data
            supply_chain_data = list(self.mongo_manager.db.supply_chain_data.find())
            if supply_chain_data:
                datasets['supply_chain'] = pd.DataFrame(supply_chain_data)
                datasets['supply_chain']['timestamp'] = pd.to_datetime(datasets['supply_chain']['timestamp'])
            
            # Load Financial Data
            financial_data = list(self.mongo_manager.db.financial_data.find())
            if financial_data:
                datasets['financial'] = pd.DataFrame(financial_data)
                datasets['financial']['timestamp'] = pd.to_datetime(datasets['financial']['timestamp'])
            
            # Load Anomaly Data
            anomaly_data = list(self.mongo_manager.db.anomaly_data.find())
            if anomaly_data:
                datasets['anomaly'] = pd.DataFrame(anomaly_data)
                datasets['anomaly']['timestamp'] = pd.to_datetime(datasets['anomaly']['timestamp'])
            
            logger.info(f"✅ Loaded {len(datasets)} datasets")
            return datasets
            
        except Exception as e:
            logger.error(f"❌ Error loading data: {e}")
            return {}
    
    def create_data_overview_dashboard(self, datasets: Dict[str, pd.DataFrame]) -> go.Figure:
        """Create comprehensive data overview dashboard"""
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=[
                'Dataset Sizes', 'Data Quality Metrics',
                'Temporal Distribution', 'Data Types Distribution',
                'Missing Data Analysis', 'Value Ranges'
            ],
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "scatter"}, {"type": "pie"}],
                   [{"type": "bar"}, {"type": "box"}]]
        )
        
        # Dataset sizes
        dataset_names = list(datasets.keys())
        dataset_sizes = [len(df) for df in datasets.values()]
        
        fig.add_trace(
            go.Bar(x=dataset_names, y=dataset_sizes, name="Record Count",
                   marker_color=self.colors['primary']),
            row=1, col=1
        )
        
        # Data quality metrics
        quality_metrics = []
        for name, df in datasets.items():
            completeness = (1 - df.isnull().sum().sum() / df.size) * 100
            quality_metrics.append(completeness)
        
        fig.add_trace(
            go.Bar(x=dataset_names, y=quality_metrics, name="Completeness %",
                   marker_color=self.colors['success']),
            row=1, col=2
        )
        
        # Temporal distribution (using IoT data as example)
        if 'iot_sensor' in datasets:
            iot_df = datasets['iot_sensor']
            if 'timestamp' in iot_df.columns:
                daily_counts = iot_df.groupby(iot_df['timestamp'].dt.date).size()
                fig.add_trace(
                    go.Scatter(x=daily_counts.index, y=daily_counts.values,
                              mode='lines+markers', name="Daily IoT Records",
                              line=dict(color=self.colors['secondary'])),
                    row=2, col=1
                )
        
        # Data types distribution
        all_columns = []
        for df in datasets.values():
            all_columns.extend(df.dtypes.value_counts().index.astype(str).tolist())
        
        type_counts = pd.Series(all_columns).value_counts()
        fig.add_trace(
            go.Pie(labels=type_counts.index, values=type_counts.values,
                   name="Data Types"),
            row=2, col=2
        )
        
        # Missing data analysis
        missing_data = []
        for name, df in datasets.items():
            missing_pct = (df.isnull().sum().sum() / df.size) * 100
            missing_data.append(missing_pct)
        
        fig.add_trace(
            go.Bar(x=dataset_names, y=missing_data, name="Missing Data %",
                   marker_color=self.colors['warning']),
            row=3, col=1
        )
        
        # Value ranges (using IoT sensor values as example)
        if 'iot_sensor' in datasets:
            iot_df = datasets['iot_sensor']
            if 'value' in iot_df.columns:
                fig.add_trace(
                    go.Box(y=iot_df['value'], name="IoT Values",
                           marker_color=self.colors['info']),
                    row=3, col=2
                )
        
        fig.update_layout(
            title="Enterprise Data Overview Dashboard",
            height=1200,
            showlegend=True,
            template="plotly_white"
        )
        
        return fig
    
    def create_iot_sensor_analysis(self, iot_df: pd.DataFrame) -> go.Figure:
        """Create detailed IoT sensor data analysis"""
        if iot_df.empty:
            return go.Figure()
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Sensor Types Distribution', 'Value Distribution by Sensor Type',
                'Temporal Trends', 'Quality Score Analysis'
            ]
        )
        
        # Sensor types distribution
        sensor_counts = iot_df['sensor_type'].value_counts()
        fig.add_trace(
            go.Bar(x=sensor_counts.index, y=sensor_counts.values,
                   name="Sensor Count", marker_color=self.colors['primary']),
            row=1, col=1
        )
        
        # Value distribution by sensor type
        for sensor_type in iot_df['sensor_type'].unique()[:5]:  # Top 5 sensor types
            sensor_data = iot_df[iot_df['sensor_type'] == sensor_type]['value']
            fig.add_trace(
                go.Box(y=sensor_data, name=sensor_type, showlegend=False),
                row=1, col=2
            )
        
        # Temporal trends
        if 'timestamp' in iot_df.columns:
            iot_df['hour'] = iot_df['timestamp'].dt.hour
            hourly_avg = iot_df.groupby('hour')['value'].mean()
            fig.add_trace(
                go.Scatter(x=hourly_avg.index, y=hourly_avg.values,
                          mode='lines+markers', name="Hourly Average",
                          line=dict(color=self.colors['secondary'])),
                row=2, col=1
            )
        
        # Quality score analysis
        if 'quality_score' in iot_df.columns:
            quality_bins = pd.cut(iot_df['quality_score'], bins=10)
            quality_counts = quality_bins.value_counts().sort_index()
            fig.add_trace(
                go.Bar(x=[str(interval) for interval in quality_counts.index],
                       y=quality_counts.values, name="Quality Distribution",
                       marker_color=self.colors['success']),
                row=2, col=2
            )
        
        fig.update_layout(
            title="IoT Sensor Data Analysis",
            height=800,
            showlegend=True,
            template="plotly_white"
        )
        
        return fig
    
    def create_supply_chain_analysis(self, supply_df: pd.DataFrame) -> go.Figure:
        """Create supply chain data analysis"""
        if supply_df.empty:
            return go.Figure()
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Supply Chain Status Distribution', 'Transportation Modes',
                'Risk Score Analysis', 'Delivery Performance'
            ]
        )
        
        # Status distribution
        status_counts = supply_df['status'].value_counts()
        fig.add_trace(
            go.Bar(x=status_counts.index, y=status_counts.values,
                   name="Status Count", marker_color=self.colors['primary']),
            row=1, col=1
        )
        
        # Transportation modes bar chart (replacing pie chart for subplot compatibility)
        transport_counts = supply_df['transportation_mode'].value_counts()
        fig.add_trace(
            go.Bar(x=transport_counts.index, y=transport_counts.values,
                   name="Transportation Modes", marker_color=self.colors['secondary']),
            row=1, col=2
        )
        
        # Risk score analysis
        if 'risk_score' in supply_df.columns:
            fig.add_trace(
                go.Histogram(x=supply_df['risk_score'], nbinsx=20,
                           name="Risk Score Distribution",
                           marker_color=self.colors['warning']),
                row=2, col=1
            )
        
        # Delivery performance
        if 'estimated_delivery' in supply_df.columns and 'actual_delivery' in supply_df.columns:
            supply_df['delivery_delay'] = (
                pd.to_datetime(supply_df['actual_delivery']) - 
                pd.to_datetime(supply_df['estimated_delivery'])
            ).dt.days
            
            delay_counts = supply_df['delivery_delay'].value_counts().head(10)
            fig.add_trace(
                go.Bar(x=delay_counts.index, y=delay_counts.values,
                       name="Delivery Delay (Days)",
                       marker_color=self.colors['info']),
                row=2, col=2
            )
        
        fig.update_layout(
            title="Supply Chain Data Analysis",
            height=800,
            showlegend=True,
            template="plotly_white"
        )
        
        return fig
    
    def create_financial_analysis(self, financial_df: pd.DataFrame) -> go.Figure:
        """Create financial data analysis"""
        if financial_df.empty:
            return go.Figure()
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Crop Price Distribution', 'Market Type Analysis',
                'Profit Margin Analysis', 'Price Volatility Trends'
            ]
        )
        
        # Crop price distribution
        if 'market_price' in financial_df.columns:
            fig.add_trace(
                go.Histogram(x=financial_df['market_price'], nbinsx=20,
                           name="Price Distribution",
                           marker_color=self.colors['primary']),
                row=1, col=1
            )
        
        # Market type analysis
        market_counts = financial_df['market_type'].value_counts()
        fig.add_trace(
            go.Pie(labels=market_counts.index, values=market_counts.values,
                   name="Market Types"),
            row=1, col=2
        )
        
        # Profit margin analysis
        if 'profit_margin' in financial_df.columns:
            fig.add_trace(
                go.Box(y=financial_df['profit_margin'], name="Profit Margins",
                       marker_color=self.colors['success']),
                row=2, col=1
            )
        
        # Price volatility trends
        if 'price_volatility' in financial_df.columns and 'timestamp' in financial_df.columns:
            financial_df['date'] = financial_df['timestamp'].dt.date
            volatility_trend = financial_df.groupby('date')['price_volatility'].mean()
            fig.add_trace(
                go.Scatter(x=volatility_trend.index, y=volatility_trend.values,
                          mode='lines+markers', name="Volatility Trend",
                          line=dict(color=self.colors['warning'])),
                row=2, col=2
            )
        
        fig.update_layout(
            title="Financial Data Analysis",
            height=800,
            showlegend=True,
            template="plotly_white"
        )
        
        return fig
    
    def create_anomaly_analysis(self, anomaly_df: pd.DataFrame) -> go.Figure:
        """Create anomaly data analysis"""
        if anomaly_df.empty:
            return go.Figure()
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Anomaly Types Distribution', 'Severity Levels',
                'Confidence Score Analysis', 'Resolution Status'
            ]
        )
        
        # Anomaly types distribution
        anomaly_counts = anomaly_df['anomaly_type'].value_counts()
        fig.add_trace(
            go.Bar(x=anomaly_counts.index, y=anomaly_counts.values,
                   name="Anomaly Count", marker_color=self.colors['warning']),
            row=1, col=1
        )
        
        # Severity levels
        severity_counts = anomaly_df['severity'].value_counts()
        fig.add_trace(
            go.Pie(labels=severity_counts.index, values=severity_counts.values,
                   name="Severity Levels"),
            row=1, col=2
        )
        
        # Confidence score analysis
        if 'confidence_score' in anomaly_df.columns:
            fig.add_trace(
                go.Histogram(x=anomaly_df['confidence_score'], nbinsx=20,
                           name="Confidence Distribution",
                           marker_color=self.colors['info']),
                row=2, col=1
            )
        
        # Resolution status
        if 'resolved' in anomaly_df.columns:
            resolved_counts = anomaly_df['resolved'].value_counts()
            fig.add_trace(
                go.Bar(x=['Unresolved', 'Resolved'], y=[resolved_counts.get(False, 0), resolved_counts.get(True, 0)],
                       name="Resolution Status", marker_color=self.colors['success']),
                row=2, col=2
            )
        
        fig.update_layout(
            title="Anomaly Data Analysis",
            height=800,
            showlegend=True,
            template="plotly_white"
        )
        
        return fig
    
    def generate_data_insights_report(self, datasets: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Generate comprehensive data insights report"""
        insights = {
            "summary": {},
            "quality_metrics": {},
            "business_insights": {},
            "recommendations": []
        }
        
        # Summary statistics
        total_records = sum(len(df) for df in datasets.values())
        insights["summary"] = {
            "total_datasets": len(datasets),
            "total_records": total_records,
            "data_types": list(datasets.keys()),
            "date_range": self._get_date_range(datasets)
        }
        
        # Quality metrics
        for name, df in datasets.items():
            completeness = (1 - df.isnull().sum().sum() / df.size) * 100
            insights["quality_metrics"][name] = {
                "completeness": round(completeness, 2),
                "total_records": len(df),
                "columns": len(df.columns),
                "memory_usage": f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB"
            }
        
        # Business insights
        insights["business_insights"] = self._generate_business_insights(datasets)
        
        # Recommendations
        insights["recommendations"] = self._generate_recommendations(datasets)
        
        return insights
    
    def _get_date_range(self, datasets: Dict[str, pd.DataFrame]) -> Dict[str, str]:
        """Get date range across all datasets"""
        all_dates = []
        for df in datasets.values():
            if 'timestamp' in df.columns:
                all_dates.extend(df['timestamp'].dropna().tolist())
        
        if all_dates:
            min_date = min(all_dates)
            max_date = max(all_dates)
            return {
                "start": str(min_date),
                "end": str(max_date),
                "span_days": (max_date - min_date).days
            }
        return {"start": "N/A", "end": "N/A", "span_days": 0}
    
    def _generate_business_insights(self, datasets: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Generate business insights from data"""
        insights = {}
        
        # IoT Sensor Insights
        if 'iot_sensor' in datasets:
            iot_df = datasets['iot_sensor']
            if 'quality_score' in iot_df.columns:
                avg_quality = iot_df['quality_score'].mean()
                insights["sensor_quality"] = {
                    "average_quality": round(avg_quality, 3),
                    "high_quality_percentage": round((iot_df['quality_score'] > 0.8).mean() * 100, 2)
                }
        
        # Supply Chain Insights
        if 'supply_chain' in datasets:
            sc_df = datasets['supply_chain']
            if 'status' in sc_df.columns:
                delivery_rate = (sc_df['status'] == 'delivered').mean() * 100
                insights["supply_chain"] = {
                    "delivery_success_rate": round(delivery_rate, 2),
                    "total_shipments": len(sc_df)
                }
        
        # Financial Insights
        if 'financial' in datasets:
            fin_df = datasets['financial']
            if 'market_price' in fin_df.columns:
                avg_price = fin_df['market_price'].mean()
                price_std = fin_df['market_price'].std()
                insights["financial"] = {
                    "average_market_price": round(avg_price, 2),
                    "price_volatility": round(price_std, 2)
                }
        
        return insights
    
    def _generate_recommendations(self, datasets: Dict[str, pd.DataFrame]) -> List[str]:
        """Generate data-driven recommendations"""
        recommendations = []
        
        # Data quality recommendations
        for name, df in datasets.items():
            completeness = (1 - df.isnull().sum().sum() / df.size) * 100
            if completeness < 95:
                recommendations.append(f"Improve data completeness for {name} dataset (currently {completeness:.1f}%)")
        
        # Business recommendations
        if 'anomaly' in datasets:
            anomaly_df = datasets['anomaly']
            if 'resolved' in anomaly_df.columns:
                resolution_rate = anomaly_df['resolved'].mean() * 100
                if resolution_rate < 80:
                    recommendations.append(f"Improve anomaly resolution rate (currently {resolution_rate:.1f}%)")
        
        if 'supply_chain' in datasets:
            sc_df = datasets['supply_chain']
            if 'status' in sc_df.columns:
                delivery_rate = (sc_df['status'] == 'delivered').mean() * 100
                if delivery_rate < 90:
                    recommendations.append(f"Improve supply chain delivery rate (currently {delivery_rate:.1f}%)")
        
        return recommendations
