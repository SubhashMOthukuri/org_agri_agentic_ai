#!/usr/bin/env python3
"""
Backend Runner - Organic Agriculture Agentic AI

Script to run the FastAPI backend with proper configuration and logging.

Usage:
    python run_backend.py [--env development|production|testing] [--port 8000] [--host 0.0.0.0]

Author: Principal AI Engineer
Version: 1.0.0
Date: December 2024
"""

import argparse
import os
import sys
import uvicorn
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backend.config import get_environment_settings, validate_config, get_settings


def main():
    """Main function to run the backend"""
    parser = argparse.ArgumentParser(
        description="Run the Organic Agriculture Agentic AI Backend",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_backend.py                           # Run in development mode
    python run_backend.py --env production         # Run in production mode
    python run_backend.py --port 8080              # Run on port 8080
    python run_backend.py --host 127.0.0.1         # Run on localhost only
        """
    )
    
    parser.add_argument(
        "--env",
        choices=["development", "production", "testing"],
        default="development",
        help="Environment to run in (default: development)"
    )
    
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind to (default: 8000)"
    )
    
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development"
    )
    
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes (default: 1)"
    )
    
    parser.add_argument(
        "--log-level",
        choices=["debug", "info", "warning", "error", "critical"],
        default="info",
        help="Log level (default: info)"
    )
    
    parser.add_argument(
        "--validate-config",
        action="store_true",
        help="Validate configuration and exit"
    )
    
    args = parser.parse_args()
    
    try:
        # Get environment-specific settings
        settings = get_environment_settings(args.env)
        
        # Override with command line arguments
        settings.host = args.host
        settings.port = args.port
        settings.debug = args.reload or settings.debug
        settings.log_level = args.log_level.upper()
        
        # Validate configuration
        print(f"🌾 Starting Organic Agriculture Agentic AI Backend")
        print(f"Environment: {args.env}")
        print(f"Host: {settings.host}")
        print(f"Port: {settings.port}")
        print(f"Debug: {settings.debug}")
        print(f"Log Level: {settings.log_level}")
        print()
        
        if args.validate_config:
            validate_config()
            print("✅ Configuration validation passed")
            return
        
        # Create logs directory
        os.makedirs("logs", exist_ok=True)
        
        # Run the application
        uvicorn.run(
            "backend.main:app",
            host=settings.host,
            port=settings.port,
            reload=settings.debug,
            workers=args.workers if not settings.debug else 1,
            log_level=settings.log_level.lower(),
            access_log=True,
            use_colors=True
        )
        
    except KeyboardInterrupt:
        print("\n🛑 Backend stopped by user")
    except Exception as e:
        print(f"❌ Error starting backend: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
