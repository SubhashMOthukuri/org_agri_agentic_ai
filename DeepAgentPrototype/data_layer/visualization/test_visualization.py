#!/usr/bin/env python3
"""
Test Data Visualization Framework

Quick test to verify visualization components work correctly.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, project_root)

def test_imports():
    """Test visualization imports"""
    try:
        print("🧪 Testing visualization imports...")
        
        # Test data explorer
        from data_layer.visualization.data_explorer import DataExplorer
        print("✅ DataExplorer import successful")
        
        # Test interactive dashboard
        from data_layer.visualization.interactive_dashboard import InteractiveDashboard
        print("✅ InteractiveDashboard import successful")
        
        # Test plotly
        import plotly.graph_objects as go
        import plotly.express as px
        print("✅ Plotly imports successful")
        
        # Test streamlit
        import streamlit as st
        print("✅ Streamlit import successful")
        
        print("🎉 All visualization imports successful!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_data_explorer():
    """Test data explorer functionality"""
    try:
        print("\n🧪 Testing DataExplorer functionality...")
        
        from data_layer.visualization.data_explorer import DataExplorer
        import pandas as pd
        import numpy as np
        
        # Create sample data
        sample_data = {
            'iot_sensor': pd.DataFrame({
                'sensor_id': ['sensor_001', 'sensor_002', 'sensor_003'],
                'farm_id': ['farm_001', 'farm_001', 'farm_002'],
                'timestamp': pd.date_range('2024-01-01', periods=3),
                'sensor_type': ['temperature', 'humidity', 'temperature'],
                'value': [25.5, 60.2, 23.1],
                'quality_score': [0.95, 0.88, 0.92]
            })
        }
        
        # Initialize data explorer
        explorer = DataExplorer()
        
        # Test dashboard creation
        fig = explorer.create_data_overview_dashboard(sample_data)
        print("✅ Data overview dashboard created")
        
        # Test insights generation
        insights = explorer.generate_data_insights_report(sample_data)
        print("✅ Data insights generated")
        
        print("🎉 DataExplorer functionality test successful!")
        return True
        
    except Exception as e:
        print(f"❌ DataExplorer test error: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 Testing Data Visualization Framework")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_data_explorer
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Visualization framework is ready.")
        return True
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
