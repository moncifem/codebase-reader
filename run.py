#!/usr/bin/env python3
"""
Simple script to run the Codebase Reader application.
This script handles dependency checking and launches Streamlit.
"""

import sys
import subprocess
import os
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    print(f"✅ Python {sys.version.split()[0]} is compatible")
    return True

def check_dependencies():
    """Check if main dependencies are installed."""
    required_packages = [
        'streamlit',
        'chromadb', 
        'sentence_transformers',
        'PyYAML',
        'python-dotenv'
    ]
    
    missing = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package} is installed")
        except ImportError:
            missing.append(package)
            print(f"❌ {package} is not installed")
    
    if missing:
        print(f"\n❌ Missing dependencies: {', '.join(missing)}")
        print("Run: pip install -r requirements.txt")
        return False
    
    return True

def check_config():
    """Check if configuration file exists."""
    if not os.path.exists('config.yaml'):
        print("❌ config.yaml not found")
        return False
    print("✅ config.yaml found")
    return True

def run_streamlit():
    """Launch the Streamlit application."""
    try:
        print("\n🚀 Starting Codebase Reader...")
        print("🌐 Open your browser and go to: http://localhost:8501")
        print("📱 To stop the application, press Ctrl+C")
        print("\n" + "="*50)
        
        # Run streamlit
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"], check=True)
        
    except KeyboardInterrupt:
        print("\n👋 Application stopped")
    except subprocess.CalledProcessError:
        print("❌ Failed to start Streamlit")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    
    return True

def main():
    """Main function."""
    print("🔍 Codebase Reader - Startup Script")
    print("="*40)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Check dependencies
    print("\nChecking dependencies...")
    if not check_dependencies():
        print("\nInstall missing dependencies and try again.")
        sys.exit(1)
    
    # Check configuration
    print("\nChecking configuration...")
    if not check_config():
        sys.exit(1)
    
    # Run basic tests if requested
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        print("\nRunning basic tests...")
        try:
            import test_basic
            test_basic.main()
        except ImportError:
            print("❌ test_basic.py not found")
        return
    
    # Launch Streamlit
    run_streamlit()

if __name__ == "__main__":
    main() 