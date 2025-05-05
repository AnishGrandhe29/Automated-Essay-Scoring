import os
import re
import pickle
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from django.conf import settings
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
# In evaluator/ml_model/analyzer.py
# Add this import at the top
from evaluator.ml_model.custom_classes import MockScaler

# Cache model and related objects
_model = None
_tokenizer = None
_scaler = None

# Car topic specific keywords and terms
CAR_KEYWORDS = [
    'car', 'automobile', 'vehicle', 'engine', 'transmission', 'horsepower', 'torque',
    'sedan', 'suv', 'coupe', 'hatchback', 'convertible', 'wagon', 'truck', 'van',
    'electric', 'hybrid', 'gasoline', 'diesel', 'combustion', 'battery', 'charging',
    'brake', 'suspension', 'steering', 'wheel', 'tire', 'safety', 'airbag',
    'toyota', 'honda', 'ford', 'chevrolet', 'bmw', 'mercedes', 'audi', 'tesla',
    'volkswagen', 'hyundai', 'kia', 'nissan', 'mazda', 'subaru', 'lexus', 'porsche',
    'fuel', 'efficiency', 'emission', 'autonomous', 'self-driving', 'driver',
    'highway', 'road', 'traffic', 'mph', 'km/h', 'acceleration', 'performance'
]

CAR_TOPIC_CATEGORIES = {
    'technical': ['engine', 'transmission', 'horsepower', 'torque', 'combustion', 'turbo', 'cylinder', 
                 'fuel', 'efficiency', 'brake', 'suspension', 'electrical', 'battery'],
    'models': ['sedan', 'suv', 'coupe', 'hatchback', 'convertible', 'wagon', 'truck', 'van', 'crossover'],
    'brands': ['toyota', 'honda', 'ford', 'chevrolet', 'bmw', 'mercedes', 'audi', 'tesla',
              'volkswagen', 'hyundai', 'kia', 'nissan', 'mazda', 'subaru', 'lexus', 'porsche'],
    'environmental': ['electric', 'hybrid', 'emissions', 'carbon', 'eco', 'environment', 'sustainable'],
    'safety': ['safety', 'airbag', 'crash', 'collision', 'autonomous', 'assist', 'braking'],
    'performance': ['speed', 'acceleration', 'handling', 'performance', 'racing', 'lap', 'track']
}

# In evaluator/ml_model/analyzer.py

# Replace your current load_model function with this:
def load_model():
    """Load the model and related objects (tokenizer, scaler) if not already loaded"""
    global _model, _tokenizer, _scaler
    
    # If model is already loaded, return
    if _model is not None:
        return _model, _tokenizer, _scaler
    
    try:
        # Check if model exists
        if not os.path.exists(settings.ML_MODEL_FILE):
            raise FileNotFoundError(f"Model file not found at {settings.ML_MODEL_FILE}")
        
        # Load model
        _model = tf.keras.models.load_model(settings.ML_MODEL_FILE)
        
        # Load tokenizer
        with open(settings.TOKENIZER_FILE, 'rb') as handle:
            _tokenizer = pickle.load(handle)
        
        # Create a simple scaler instead of loading from pickle
        class SimpleScaler:
            def transform(self, X):
                # Just return the same data standardized 
                X = np.array(X)
                return (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-8)
        
        _scaler = SimpleScaler()
        
        return _model, _tokenizer, _scaler
        
    except Exception as e:
        # If there's an error, set the objects to None and re-raise
        _model, _tokenizer, _scaler = None, None, None
        raise Exception(f"Error loading model: {str(e)}")

def is_car_topic(text, threshold=0.01):
    """
    Determine if an essay is primarily about cars based on keyword density.
    
    Parameters:
        text: The essay text
        threshold: Minimum keyword density to classify as car topic
    
    Returns:
        Boolean indicating if essay is about cars and keyword density
    """
    text_lower = text.lower()
    word_count = len(text_lower.split())
    
    # Count occurrences of car-related keywords
    car_word_count = 0
    for keyword in CAR_KEYWORDS:
        # Count full word matches (not partial)
        pattern = r'\b' + re.escape(keyword) + r'\b'
        car_word_count += len(re.findall(pattern, text_lower))
    
    # Calculate keyword density
    keyword_density = car_word_count / word_count if word_count > 0 else 0
    
    return keyword_density >= threshold, keyword_density

def analyze_car_topic_coverage(text):
    """
    Analyze how well the essay covers different aspects of car topics.
    
    Parameters:
        text: The essay text
    
    Returns:
        Dictionary with coverage scores for different car-related categories
    """
    text_lower = text.lower()
    coverage = {}
    
    for category, keywords in CAR_TOPIC_CATEGORIES.items():
        category_hits = 0
        for keyword in keywords:
            pattern = r'\b' + re.escape(keyword) + r'\b'
            category_hits += len(re.findall(pattern, text_lower))
        
        coverage[category] = category_hits
    
    return coverage

def preprocess_text(text):
    """Clean and preprocess the essay text."""
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Remove extra whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_car_topic_features(text):
    """Extract features specifically related to car topics."""
    text_lower = text.lower()
    is_car_essay, keyword_density = is_car_topic(text_lower)
    
    # Get coverage across different car topic categories
    coverage = analyze_car_topic_coverage(text_lower)
    
    # Calculate diversity of car topics (how many different categories are covered)
    categories_covered = sum(1 for category, hits in coverage.items() if hits > 0)
    coverage_ratio = categories_covered / len(CAR_TOPIC_CATEGORIES)
    
    # Technical accuracy estimation (simplified approach)
    # Higher hit count in technical category might suggest better technical knowledge
    technical_depth = coverage.get('technical', 0) / 5 if coverage.get('technical', 0) > 0 else 0
    
    # Domain-specific analysis
    if is_car_essay:
        # Create features specific to car essays
        result = {
            'is_car_topic': 1,
            'car_keyword_density': keyword_density,
            'car_topic_coverage': coverage_ratio,
            'technical_depth': min(technical_depth, 1.0)  # Cap at 1.0
        }
        
        # Add individual category coverage
        for category, hits in coverage.items():
            result[f'category_{category}'] = hits
            
        return result
    else:
        # Return default values if not a car essay
        default_result = {
            'is_car_topic': 0,
            'car_keyword_density': 0,
            'car_topic_coverage': 0,
            'technical_depth': 0
        }
        
        # Add zeroes for individual categories
        for category in CAR_TOPIC_CATEGORIES.keys():
            default_result[f'category_{category}'] = 0
            
        return default_result

def extract_coherence_features(text):
    """Extract features related to essay coherence."""
    try:
        # Using NLTK's sentence tokenizer
        sentences = sent_tokenize(text)
        
        # Calculate sentence length variation (might indicate coherence)
        sent_lengths = [len(word_tokenize(sent)) for sent in sentences]
        sent_length_var = np.var(sent_lengths) if len(sent_lengths) > 1 else 0
        
        # Calculate average and max sentence length
        avg_sent_length = np.mean(sent_lengths) if sent_lengths else 0
        max_sent_length = max(sent_lengths) if sent_lengths else 0
        
        return {
            'sent_length_var': sent_length_var,
            'avg_sent_length': avg_sent_length,
            'max_sent_length': max_sent_length,
            'num_sentences': len(sentences)
        }
    except Exception as e:
        # Fallback if tokenization fails
        print(f"Error in coherence extraction: {e}")
        return {
            'sent_length_var': 0.0,
            'avg_sent_length': 0.0,
            'max_sent_length': 0.0,
            'num_sentences': 1  # Assume at least one sentence
        }

def calculate_transition_words(text):
    """Count transition words/phrases to measure flow between sentences."""
    transition_words = [
        'however', 'therefore', 'furthermore', 'moreover', 'in addition',
        'consequently', 'as a result', 'thus', 'instead', 'nevertheless',
        'for example', 'specifically', 'in contrast', 'similarly', 'on the other hand'
    ]
    
    word_count = 0
    text_lower = text.lower()
    
    for word in transition_words:
        word_count += text_lower.count(word)
    
    return word_count

def generate_car_essay_recommendations(coverage, essay_text):
    """Generate specific recommendations for improving a car essay."""
    recommendations = []
    
    # Check if technical information is lacking
    if coverage.get('technical', 0) <= 2:
        recommendations.append("Consider including more technical details about car specifications, engine types, or mechanical systems.")
    
    # Check if environmental aspects are missing
    if coverage.get('environmental', 0) <= 1:
        recommendations.append("The essay could benefit from discussing environmental impacts of cars, such as emissions or alternative fuel technologies.")
    
    # Check if safety aspects are missing
    if coverage.get('safety', 0) <= 1:
        recommendations.append("Consider adding information about car safety features, crash ratings, or safety technologies.")
    
    # Check if the essay is too brand-focused without technical content
    if coverage.get('brands', 0) > 5 and coverage.get('technical', 0) <= 2:
        recommendations.append("The essay mentions many car brands but lacks technical depth. Consider adding more technical details to support brand comparisons.")
    
    # Add generic recommendation if no specific issues found
    if not recommendations:
        recommendations.append("The essay covers car topics well. Consider expanding on the analysis of future trends in the automotive industry.")
    
    return recommendations
def analyze_essay(essay_text, essay_features, model, tokenizer, scaler, car_features=None):
    """
    Analyze an individual essay and provide detailed feedback.
    
    Parameters:
        essay_text: The text of the essay
        essay_features: Numerical features of the essay
        model: Trained model
        tokenizer: Text tokenizer
        scaler: Feature scaler
        car_features: Optional dictionary of car-topic features (if any)
    
    Returns:
        Dictionary with analysis results
    """
    try:
        # Preprocess the text
        processed_text = preprocess_text(essay_text)
        
        # Convert to sequence
        sequence = tokenizer.texts_to_sequences([processed_text])
        padded_sequence = pad_sequences(sequence, maxlen=500, padding='post', truncating='post')
        
        # Append car features if provided
        full_features = essay_features.copy()
        if car_features:
            for feature_name in sorted(car_features.keys()):
                full_features.append(car_features[feature_name])
        else:
            full_features = essay_features
        print(len(essay_features))
        print(len(car_features))
        print(len(full_features))
        # Scale features
        scaled_feat = scaler.transform([full_features])
        
        # Predict score
        predicted_score = model.predict([padded_sequence, scaled_feat])[0][0]
        
        # Analyze strengths and weaknesses
        analysis = {}
        analysis['predicted_score'] = predicted_score
        
        # Analyze sentence structure
        sentences = sent_tokenize(essay_text)
        sent_lengths = [len(word_tokenize(sent)) for sent in sentences]
        
        analysis['sentence_analysis'] = {
            'num_sentences': len(sentences),
            'avg_sentence_length': np.mean(sent_lengths) if sent_lengths else 0,
            'sentence_length_variation': np.var(sent_lengths) if len(sent_lengths) > 1 else 0
        }
        
        # Vocabulary richness analysis
        words = word_tokenize(processed_text)
        unique_words = set(words)
        
        analysis['vocabulary_analysis'] = {
            'total_words': len(words),
            'unique_words': len(unique_words),
            'lexical_diversity': len(unique_words) / len(words) if words else 0
        }
        
        # POS tag analysis
        pos_tags = nltk.pos_tag(words)
        tag_counts = {}
        for _, tag in pos_tags:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1
        
        analysis['pos_tag_analysis'] = tag_counts
        
        # Transition words analysis
        transition_count = calculate_transition_words(essay_text)
        analysis['transition_words'] = transition_count
        analysis['transition_density'] = transition_count / len(words) if words else 0

        # Include car-topic breakdown if present
        if car_features:
            analysis['car_topic_features'] = car_features
        
        return analysis

    except Exception as e:
        print(f"Error analyzing essay: {e}")
        return {
            'error': str(e),
            'message': "An error occurred while analyzing the essay."
        }
def analyze_car_essay(essay_text, essay_features, car_features, model, tokenizer, scaler):
    """
    Analyze an individual car essay and provide detailed feedback.
    
    Parameters:
        essay_text: The text of the essay
        essay_features: General numerical features of the essay
        car_features: Car-specific features
        model: Trained model
        tokenizer: Text tokenizer
        scaler: Feature scaler
    
    Returns:
        Dictionary with analysis results and car-specific feedback
    """
    try:
        # Get the general essay analysis first
        general_analysis = analyze_essay(essay_text, essay_features, model, tokenizer, scaler,car_features)
        
        # Add car-specific analysis
        is_car_topic_result, keyword_density = is_car_topic(essay_text)
        
        if not is_car_topic_result:
            general_analysis['car_relevance'] = {
                'is_car_topic': False,
                'car_keyword_density': keyword_density,
                'recommendation': 'This essay does not appear to be primarily about cars. Consider adding more car-related content.'
            }
            return general_analysis
            
        # Get topic coverage
        coverage = analyze_car_topic_coverage(essay_text)
        
        # Identify strengths and weaknesses in topic coverage
        strongest_category = max(coverage.items(), key=lambda x: x[1])[0] if coverage else None
        weakest_categories = [cat for cat, hits in coverage.items() if hits <= 1]
        
        # Calculate car topic specific score adjustment
        # This is a simplified approach - in a real model you'd train on car essays
        coverage_score = sum(coverage.values()) / (len(CAR_TOPIC_CATEGORIES) * 5)  # Normalize to 0-1
        technical_accuracy = coverage.get('technical', 0) / 10  # Simplified technical accuracy score
        
        # Weight the original score and the car-specific score
        # In a real implementation, you would train the model on car essays
        car_specific_score = (general_analysis['predicted_score'] * 0.7) + \
                           (coverage_score * 0.2) + \
                           (technical_accuracy * 0.1)
        
        # Add car-specific analysis to the results
        general_analysis['car_topic_analysis'] = {
            'is_car_topic': True,
            'car_keyword_density': keyword_density,
            'car_specific_score': car_specific_score,
            'coverage_by_category': coverage,
            'strongest_category': strongest_category,
            'areas_for_improvement': weakest_categories if weakest_categories else ["No specific weak areas identified"],
            'recommendations': generate_car_essay_recommendations(coverage, essay_text)
        }
        
        return general_analysis
    
    except Exception as e:
        print(f"Error analyzing car essay: {e}")
        return {
            'predicted_score': 0,
            'error': str(e),
            'car_topic_analysis': {
                'is_car_topic': False,
                'car_keyword_density': 0,
                'error': 'Analysis failed'
            }
        }
def score_car_essay(essay_text):
    """
    Score a new car essay using the trained model.
    
    Parameters:
        essay_text: The text of the car essay
    
    Returns:
        Dictionary with score and analysis
    """
    try:
        # Load model and artifacts
        model, tokenizer, scaler = load_model()
        
        # Check if the essay is about cars
        is_car_topic_result, keyword_density = is_car_topic(essay_text)
        
        if not is_car_topic_result:
            return {
                'score': 0,
                'is_car_topic': False,
                'car_keyword_density': keyword_density,
                'message': "This essay does not appear to be primarily about cars. Please submit an essay focused on automotive topics."
            }
        
        print("Hi1")
        
        # Process the essay
        processed_text = preprocess_text(essay_text)
        
        # Extract features
        coherence = extract_coherence_features(essay_text)
        car_features = extract_car_topic_features(essay_text)
        transition_words = calculate_transition_words(essay_text)
        
        # Count words, characters, sentences
        words = word_tokenize(processed_text)
        word_count = len(words)
        char_count = len(processed_text)
        avg_word_length = char_count / word_count if word_count > 0 else 0
        sentence_count = len(sent_tokenize(essay_text))
        
        # Simple POS tagging for noun, verb, adjective, adverb counts
        pos_tags = nltk.pos_tag(words)
        noun_count = sum(1 for _, tag in pos_tags if tag.startswith('NN'))
        verb_count = sum(1 for _, tag in pos_tags if tag.startswith('VB'))
        adj_count = sum(1 for _, tag in pos_tags if tag.startswith('JJ'))
        adv_count = sum(1 for _, tag in pos_tags if tag.startswith('RB'))
        
        # Calculate punctuation count
        punctuation_count = len(re.findall(r'[.,!?;:]', essay_text))
        
        # Prepare features in the expected format
        essay_features = [
            word_count, char_count, avg_word_length, sentence_count, punctuation_count,
            noun_count, verb_count, adj_count, adv_count, transition_words,
            coherence['sent_length_var'], coherence['avg_sent_length'], coherence['max_sent_length'], 0
        ]
        
        # Prepare text sequence
        sequence = tokenizer.texts_to_sequences([processed_text])
        padded_sequence = pad_sequences(sequence, maxlen=500, padding='post', truncating='post')
        full_features = essay_features.copy()
        if car_features:
            for feature_name in sorted(car_features.keys()):
                full_features.append(car_features[feature_name])
        else:
            full_features = essay_features
        print(len(essay_features))
        print(len(car_features))
        print(len(full_features))
        # Scale features
        print("Hi2")
        scaled_feat = scaler.transform([full_features])
        

        print("hi3")
        
        # Predict score
        predicted_score = model.predict([padded_sequence, scaled_feat])[0][0]

        print("hi5")
        
        # Perform detailed analysis
        car_analysis = analyze_car_essay(essay_text, 
                                         essay_features,  # Original features
                                         car_features, 
                                         model, 
                                         tokenizer, 
                                         scaler)
        print("hi4")
        
        return {
            'score': predicted_score,
            'is_car_topic': True,
            'car_keyword_density': keyword_density,
            'analysis': car_analysis
        }
        
    except Exception as e:
        print(f"Error scoring car essay: {e}")
        return {
            'score': 0,
            'error': str(e),
            'message': "An error occurred while scoring the essay."
        }