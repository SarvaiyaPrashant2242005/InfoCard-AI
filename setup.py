"""
Setup script for easy installation of the Business Card Scanner application.
Run: python setup.py install
"""

import os
import subprocess
import sys
from pathlib import Path

def check_python_version():
    """Check if Python version is 3.7+"""
    if sys.version_info < (3, 7):
        print("Error: Python 3.7 or higher is required.")
        sys.exit(1)
    print("✓ Python version check passed")

def create_structure():
    """Create necessary directories"""
    directories = ["uploads", "templates"]
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
    print("✓ Directory structure created")

def install_requirements():
    """Install required packages"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ Dependencies installed")
    except subprocess.CalledProcessError:
        print("Error: Failed to install requirements. Please try manually: pip install -r requirements.txt")
        sys.exit(1)

def install_spacy_model():
    """Install spaCy English model"""
    try:
        subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
        print("✓ spaCy English model installed")
    except subprocess.CalledProcessError:
        print("Error: Failed to install spaCy model. Please try manually: python -m spacy download en_core_web_sm")
        sys.exit(1)

def verify_files():
    """Verify that all necessary files exist"""
    required_files = [
        "app.py",
        "templates/index.html",
        "templates/view.html",
        "requirements.txt"
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print(f"Error: Missing files: {', '.join(missing_files)}")
        sys.exit(1)
    
    print("✓ All required files are present")

def main():
    """Main setup function"""
    print("Setting up Business Card Scanner application...")
    
    check_python_version()
    create_structure()
    verify_files()
    install_requirements()
    install_spacy_model()
    
    print("\nSetup completed successfully!")
    print("To run the application: python app.py")
    print("Then open your browser to: http://127.0.0.1:5000/")

if __name__ == "__main__":
    main()