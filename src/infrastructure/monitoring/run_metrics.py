#!/usr/bin/env python3
"""
Metrics Runner - Run and display project metrics

This script runs the comprehensive metrics system and displays
all project metrics in an organized dashboard format.

Usage:
    python run_metrics.py [--format json|csv|dashboard] [--export] [--config]

Author: Principal AI Engineer
Version: 1.0.0
Date: December 2024
"""

import argparse
import sys
import os
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from metrics.project_metrics import ProjectMetrics, create_metrics_dashboard
from metrics.metrics_config import get_metrics_config


def main():
    """Main function to run metrics"""
    parser = argparse.ArgumentParser(
        description="Run and display project metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_metrics.py                    # Display dashboard
    python run_metrics.py --format json      # Export as JSON
    python run_metrics.py --format csv       # Export as CSV
    python run_metrics.py --export           # Export all formats
    python run_metrics.py --config           # Show configuration
        """
    )
    
    parser.add_argument(
        "--format", 
        choices=["json", "csv", "dashboard"], 
        default="dashboard",
        help="Output format (default: dashboard)"
    )
    
    parser.add_argument(
        "--export", 
        action="store_true",
        help="Export metrics to files"
    )
    
    parser.add_argument(
        "--config", 
        action="store_true",
        help="Show metrics configuration"
    )
    
    parser.add_argument(
        "--output-dir", 
        default="metrics/exports",
        help="Output directory for exports (default: metrics/exports)"
    )
    
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        if args.config:
            show_configuration()
            return
        
        # Initialize metrics
        metrics = ProjectMetrics()
        metrics.load_enterprise_metrics()
        
        if args.format == "dashboard":
            display_dashboard(metrics, args.verbose)
        elif args.format == "json":
            export_json(metrics, output_dir, args.verbose)
        elif args.format == "csv":
            export_csv(metrics, output_dir, args.verbose)
        
        if args.export:
            export_all_formats(metrics, output_dir, args.verbose)
            
    except Exception as e:
        print(f"❌ Error running metrics: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


def show_configuration():
    """Show metrics configuration"""
    config = get_metrics_config()
    summary = config.get_config_summary()
    
    print("=" * 80)
    print("🌾 METRICS CONFIGURATION")
    print("=" * 80)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    print("📊 METRICS OVERVIEW")
    print("-" * 40)
    print(f"Total Metrics: {summary['total_metrics']}")
    print(f"Enabled Metrics: {summary['enabled_metrics']}")
    print(f"Disabled Metrics: {summary['disabled_metrics']}")
    print()
    
    print("🚨 ALERT LEVELS")
    print("-" * 40)
    for level, count in summary['alert_levels'].items():
        print(f"{level.upper()}: {count} metrics")
    print()
    
    print("⚙️ COLLECTION SETTINGS")
    print("-" * 40)
    for key, value in summary['collection_settings'].items():
        print(f"{key}: {value}")
    print()
    
    print("📁 EXPORT SETTINGS")
    print("-" * 40)
    for key, value in summary['export_settings'].items():
        print(f"{key}: {value}")
    print()


def display_dashboard(metrics: ProjectMetrics, verbose: bool = False):
    """Display metrics dashboard"""
    print("=" * 80)
    print("🌾 ORGANIC AGRICULTURE AGENTIC AI - METRICS DASHBOARD")
    print("=" * 80)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Overall Health: {metrics._calculate_overall_health()}")
    print()
    
    # Summary
    summary = metrics.get_metrics_summary()
    print("📊 METRICS SUMMARY")
    print("-" * 40)
    print(f"Total Metrics: {summary['total_metrics']}")
    print(f"Critical: {summary['critical_metrics']}")
    print(f"Warning: {summary['warning_metrics']}")
    print()
    
    # Category breakdown
    print("📈 METRICS BY CATEGORY")
    print("-" * 40)
    for category, stats in summary['category_summary'].items():
        print(f"{category.upper()}:")
        print(f"  Total: {stats['count']}")
        print(f"  Critical: {stats['critical']} | Warning: {stats['warning']}")
        print(f"  Good: {stats['good']} | Excellent: {stats['excellent']}")
        print()
    
    # Critical metrics
    critical_metrics = metrics.get_critical_metrics()
    if critical_metrics:
        print("🚨 CRITICAL METRICS")
        print("-" * 40)
        for metric in critical_metrics:
            print(f"• {metric.name}: {metric.value} {metric.threshold.unit} - {metric.description}")
        print()
    
    # Warning metrics
    warning_metrics = metrics.get_warning_metrics()
    if warning_metrics:
        print("⚠️  WARNING METRICS")
        print("-" * 40)
        for metric in warning_metrics:
            print(f"• {metric.name}: {metric.value} {metric.threshold.unit} - {metric.description}")
        print()
    
    # Top performing metrics
    excellent_metrics = [m for m in metrics.metrics.values() if m.status.value == "excellent"]
    if excellent_metrics:
        print("✅ EXCELLENT METRICS")
        print("-" * 40)
        for metric in excellent_metrics[:5]:  # Show top 5
            print(f"• {metric.name}: {metric.value} {metric.threshold.unit} - {metric.description}")
        print()
    
    if verbose:
        print("🔍 DETAILED METRICS")
        print("-" * 40)
        for name, metric in metrics.metrics.items():
            print(f"{name}:")
            print(f"  Value: {metric.value} {metric.threshold.unit}")
            print(f"  Status: {metric.status.value}")
            print(f"  Category: {metric.category.value}")
            print(f"  Description: {metric.description}")
            if metric.trend:
                print(f"  Trend: {metric.trend}")
            print()


def export_json(metrics: ProjectMetrics, output_dir: Path, verbose: bool = False):
    """Export metrics as JSON"""
    json_export = metrics.export_metrics("json")
    output_file = output_dir / f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(output_file, "w") as f:
        f.write(json_export)
    
    if verbose:
        print(f"✅ Metrics exported to JSON: {output_file}")
    else:
        print(f"📁 JSON export: {output_file}")


def export_csv(metrics: ProjectMetrics, output_dir: Path, verbose: bool = False):
    """Export metrics as CSV"""
    csv_export = metrics.export_metrics("csv")
    output_file = output_dir / f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    with open(output_file, "w") as f:
        f.write(csv_export)
    
    if verbose:
        print(f"✅ Metrics exported to CSV: {output_file}")
    else:
        print(f"📁 CSV export: {output_file}")


def export_all_formats(metrics: ProjectMetrics, output_dir: Path, verbose: bool = False):
    """Export metrics in all formats"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # JSON export
    json_export = metrics.export_metrics("json")
    json_file = output_dir / f"metrics_{timestamp}.json"
    with open(json_file, "w") as f:
        f.write(json_export)
    
    # CSV export
    csv_export = metrics.export_metrics("csv")
    csv_file = output_dir / f"metrics_{timestamp}.csv"
    with open(csv_file, "w") as f:
        f.write(csv_export)
    
    # Summary export
    summary = metrics.get_metrics_summary()
    summary_file = output_dir / f"metrics_summary_{timestamp}.json"
    with open(summary_file, "w") as f:
        import json
        json.dump(summary, f, indent=2, default=str)
    
    if verbose:
        print("📁 EXPORT COMPLETE")
        print("-" * 40)
        print(f"JSON: {json_file}")
        print(f"CSV: {csv_file}")
        print(f"Summary: {summary_file}")
    else:
        print(f"📁 Exported to: {output_dir}")


if __name__ == "__main__":
    main()
