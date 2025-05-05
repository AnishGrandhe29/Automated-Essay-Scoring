#!/usr/bin/env python
"""
Setup script to initialize the Car Essay Evaluation model.
This script will create mock model files if the real model files don't exist.
"""

import os
import sys
import argparse
import shutil
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional, Input


def create_directories():
    """Create necessary directories if they don't exist."""
    directories = [
        'evaluator/ml_models',
        'evaluator/static/css',
        'evaluator/static/js',
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Created directory: {directory}")

# In your setup_model.py, change the model creation part to be simpler:

def create_mock_model():
    """Create a mock model for demonstration purposes."""
    model_path = 'evaluator/ml_models'
    os.makedirs(model_path, exist_ok=True)
    
    print("Creating mock model files...")
    
    # Create mock text data
    mock_texts = [
        f"This is a mock car essay about engine performance and fuel efficiency. Cars like Toyota and Honda are mentioned. {i}" 
        for i in range(1000)
    ]
    
    # Create tokenizer
    tokenizer = Tokenizer(num_words=5000, oov_token='<OOV>')
    tokenizer.fit_on_texts(mock_texts)
    
    # Save the tokenizer
    with open(os.path.join(model_path, 'car_tokenizer.pickle'), 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"✓ Created tokenizer file: {os.path.join(model_path, 'car_tokenizer.pickle')}")
    
    # Create a dummy model file to indicate setup is complete
    with open(os.path.join(model_path, 'model_created.txt'), 'w') as f:
        f.write("Model setup complete")
    print(f"✓ Created model placeholder file")
    
    print("Mock artifacts created successfully!")
    return True

def download_nltk_resources():
    """Download required NLTK resources."""
    import nltk
    
    resources = [
        'punkt',
        'averaged_perceptron_tagger',
        'stopwords'
    ]
    
    print("Downloading NLTK resources...")
    for resource in resources:
        try:
            nltk.download(resource, quiet=True)
            print(f"✓ Downloaded NLTK resource: {resource}")
        except Exception as e:
            print(f"✗ Error downloading NLTK resource {resource}: {str(e)}")
    
    return True

def check_existing_model():
    """Check if model files already exist."""
    model_path = 'evaluator/ml_models'
    required_files = [
        'car_essay_evaluation_model.h5',
        'car_tokenizer.pickle'
    ]
    
    if not os.path.exists(model_path):
        return False
        
    for file in required_files:
        if not os.path.exists(os.path.join(model_path, file)):
            return False
            
    return True

def setup_django_app():
    """Set up Django app initial configuration."""
    try:
        # Create an empty __init__.py in migrations folder
        migrations_dir = 'evaluator/migrations'
        os.makedirs(migrations_dir, exist_ok=True)
        
        init_file = os.path.join(migrations_dir, '__init__.py')
        if not os.path.exists(init_file):
            with open(init_file, 'w') as f:
                f.write('# This file is intentionally left empty\n')
            print(f"✓ Created migrations init file: {init_file}")
                
        # Run Django migrations
        print("\nSetting up Django database...")
        os.system('python manage.py makemigrations evaluator')
        os.system('python manage.py migrate')
        
        return True
    except Exception as e:
        print(f"✗ Error setting up Django app: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Set up the Car Essay Evaluation System')
    parser.add_argument('--force', action='store_true', help='Force recreation of model files')
    args = parser.parse_args()
    
    print("\n===== Car Essay Evaluation System Setup =====\n")
    
    # Create necessary directories
    create_directories()
    
    # Check for existing model files
    if check_existing_model() and not args.force:
        print("\nModel files already exist. Use --force to recreate them.")
    else:
        # Create mock model if needed
        create_mock_model()
    
    # Download NLTK resources
    download_nltk_resources()
    
    # Set up Django app
    setup_django_app()
    
    print("\n===== Setup Complete =====")
    print("\nYou can now start the application with:")
    print("  python manage.py runserver")
    print("\nAccess the application at: http://127.0.0.1:8000/")

if __name__ == "__main__":
    main()