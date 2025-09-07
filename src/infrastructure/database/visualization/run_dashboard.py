#!/usr/bin/env python3
"""
Run Interactive Data Visualization Dashboard

Principal AI Engineer level data exploration dashboard.
"""

import subprocess
import sys
import os
from pathlib import Path

def install_requirements():
    """Install required packages for visualization"""
    packages = [
        "streamlit",
        "plotly",
        "pandas",
        "numpy",
        "pymongo"
    ]
    
    print("📦 Installing visualization dependencies...")
    for package in packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✅ Installed {package}")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {package}: {e}")

def run_dashboard():
    """Run the Streamlit dashboard"""
    dashboard_path = Path(__file__).parent / "interactive_dashboard.py"
    
    print("🚀 Starting Interactive Data Dashboard...")
    print("📊 Dashboard will open in your browser at http://localhost:8501")
    print("🛑 Press Ctrl+C to stop the dashboard")
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            str(dashboard_path),
            "--server.port", "8501",
            "--server.address", "localhost"
        ])
    except KeyboardInterrupt:
        print("\n🛑 Dashboard stopped by user")
    except Exception as e:
        print(f"❌ Error running dashboard: {e}")

def main():
    """Main function"""
    print("🌾 Organic Agriculture AI - Data Visualization Dashboard")
    print("=" * 60)
    
    # Install requirements
    install_requirements()
    
    # Run dashboard
    run_dashboard()

if __name__ == "__main__":
    main()
