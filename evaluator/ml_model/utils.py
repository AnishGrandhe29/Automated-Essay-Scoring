"""
Utility functions for the car essay evaluation model.
"""

import os
import re
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional, Input
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

def mock_train_model(save_path, data_size=1000):
    """
    Create a mock model for demonstration purposes.
    This is useful when you don't have the actual trained model.
    
    Parameters:
        save_path: Directory to save model files
        data_size: Size of mock data to create
    """
    # Create directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    
    # Create mock text data
    mock_texts = [
        f"This is a mock car essay about engine performance and fuel efficiency. Cars like Toyota and Honda are mentioned. {i}" 
        for i in range(data_size)
    ]
    
    # Create mock scores
    mock_scores = np.random.uniform(1, 10, size=data_size)
    
    # Create tokenizer
    tokenizer = Tokenizer(num_words=5000, oov_token='<OOV>')
    tokenizer.fit_on_texts(mock_texts)
    
    # Create model
    max_length = 100
    
    # Text branch (LSTM)
    text_input = Input(shape=(max_length,), name='text_input')
    embedding_layer = Embedding(5000, 64)(text_input)
    lstm_layer = Bidirectional(LSTM(32, return_sequences=True))(embedding_layer)
    lstm_layer = Bidirectional(LSTM(32))(lstm_layer)
    lstm_output = Dense(16, activation='relu')(lstm_layer)

    # Numerical features branch
    numerical_input = Input(shape=(20,), name='numerical_input')
    numerical_dense = Dense(16, activation='relu')(numerical_input)
    numerical_dense = Dense(8, activation='relu')(numerical_dense)

    # Combine both branches
    concatenated = tf.keras.layers.concatenate([lstm_output, numerical_dense])
    dense_layer = Dense(16, activation='relu')(concatenated)
    dense_layer = Dropout(0.3)(dense_layer)
    dense_layer = Dense(8, activation='relu')(dense_layer)
    output = Dense(1, activation='linear')(dense_layer)

    # Create the model
    model = Model(inputs=[text_input, numerical_input], outputs=output)

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    
    # Save the model
    model.save(os.path.join(save_path, 'car_essay_evaluation_model.h5'))
    
    # Save the tokenizer
    with open(os.path.join(save_path, 'car_tokenizer.pickle'), 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    # Create a mock scaler (simple StandardScaler-like object)
    class MockScaler:
        def transform(self, X):
            # Just return the same data standardized 
            return (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    
    mock_scaler = MockScaler()
    
    # Save the scaler
    with open(os.path.join(save_path, 'car_feature_scaler.pickle'), 'wb') as handle:
        pickle.dump(mock_scaler, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    print(f"Mock model and artifacts created and saved to {save_path}")
    return model, tokenizer, mock_scaler

def format_score_as_grade(score):
    """
    Convert a numerical score (0-10) to a letter grade with descriptive text.
    
    Parameters:
        score: Numerical score between 0 and 10
        
    Returns:
        Tuple of (letter_grade, description)
    """
    if score >= 9.5:
        return ("A+", "Outstanding")
    elif score >= 9.0:
        return ("A", "Excellent")
    elif score >= 8.5:
        return ("A-", "Very Good")
    elif score >= 8.0:
        return ("B+", "Good")
    elif score >= 7.5:
        return ("B", "Above Average")
    elif score >= 7.0:
        return ("B-", "Slightly Above Average")
    elif score >= 6.5:
        return ("C+", "Average")
    elif score >= 6.0:
        return ("C", "Satisfactory")
    elif score >= 5.5:
        return ("C-", "Below Average")
    elif score >= 5.0:
        return ("D+", "Needs Improvement")
    elif score >= 4.0:
        return ("D", "Poor")
    elif score >= 3.0:
        return ("D-", "Very Poor")
    else:
        return ("F", "Unsatisfactory")

def create_spider_chart_data(coverage_data):
    """
    Prepare car topic coverage data for a spider/radar chart.
    
    Parameters:
        coverage_data: Dictionary of category coverage counts
        
    Returns:
        Dictionary with normalized data and labels
    """
    categories = list(coverage_data.keys())
    values = list(coverage_data.values())
    
    # Normalize values to 0-100 scale
    max_value = max(values) if max(values) > 0 else 1
    normalized_values = [min(100, (val / max_value) * 100) for val in values]
    
    return {
        'categories': categories,
        'values': normalized_values
    }

def count_pos_tags(essay_text):
    """
    Count the occurrences of different parts of speech in an essay.
    
    Parameters:
        essay_text: The text of the essay
        
    Returns:
        Dictionary with counts for different POS tags
    """
    import nltk
    from nltk.tokenize import word_tokenize
    
    # Make sure necessary NLTK resources are downloaded
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('taggers/averaged_perceptron_tagger')
    except LookupError:
        nltk.download('punkt')
        nltk.download('averaged_perceptron_tagger')
    
    # Tokenize and tag the text
    tokens = word_tokenize(essay_text)
    pos_tags = nltk.pos_tag(tokens)
    
    # Group tags into major categories
    # Group tags into major categories
    pos_categories = {
        'nouns': ['NN', 'NNS', 'NNP', 'NNPS'],
        'verbs': ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'],
        'adjectives': ['JJ', 'JJR', 'JJS'],
        'adverbs': ['RB', 'RBR', 'RBS'],
        'pronouns': ['PRP', 'PRP$', 'WP', 'WP$'],
        'determiners': ['DT', 'PDT', 'WDT'],
        'conjunctions': ['CC', 'IN'],
        'others': []
    }
    
    # Count tags by category
    counts = {category: 0 for category in pos_categories}
    
    for _, tag in pos_tags:
        category_found = False
        for category, tags in pos_categories.items():
            if tag in tags:
                counts[category] += 1
                category_found = True
                break
        
        if not category_found:
            counts['others'] += 1
    
    return counts