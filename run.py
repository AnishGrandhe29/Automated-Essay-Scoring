#!/usr/bin/env python
"""
Run script for the Car Essay Evaluation System.
This script checks for the model files, sets up the environment, and runs the Django server.
"""

import os
import sys
import subprocess
import argparse

def check_environment():
    """Check if the virtual environment is active."""
    return hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)

def check_dependencies():
    """Check if dependencies are installed."""
    try:
        import django
        import tensorflow
        import nltk
        import numpy
        import pandas
        return True
    except ImportError as e:
        print(f"Missing dependency: {str(e)}")
        return False

def check_model_files():
    """Check if model files exist."""
    model_path = 'evaluator/ml_models'
    required_files = [
        'car_essay_evaluation_model.h5',
        'car_tokenizer.pickle',
        'car_feature_scaler.pickle'
    ]
    
    if not os.path.exists(model_path):
        return False
        
    for file in required_files:
        if not os.path.exists(os.path.join(model_path, file)):
            return False
            
    return True

def run_setup():
    """Run the setup script."""
    print("Running setup script...")
    subprocess.run([sys.executable, 'setup_model.py'])

def run_server(port=8000):
    """Run the Django development server."""
    print(f"\nStarting Django server on port {port}...")
    subprocess.run([sys.executable, 'manage.py', 'runserver', f'127.0.0.1:{port}'])

def main():
    parser = argparse.ArgumentParser(description='Run the Car Essay Evaluation System')
    parser.add_argument('--port', type=int, default=8000, help='Port to run the server on')
    parser.add_argument('--setup', action='store_true', help='Run setup before starting server')
    args = parser.parse_args()
    
    print("\n===== Car Essay Evaluation System =====\n")
    
    # Check if we're in a virtual environment
    if not check_environment():
        print("Warning: Virtual environment not detected. It's recommended to run this in a virtual environment.")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(0)
    
    # Check dependencies
    if not check_dependencies():
        print("\nSome dependencies are missing. Installing required packages...")
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
    
    # Run setup if requested or if model files don't exist
    if args.setup or not check_model_files():
        run_setup()
    
    # Run the server
    run_server(args.port)

if __name__ == "__main__":
    main()