from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages
from django.http import JsonResponse
from django.views.decorators.http import require_POST
from django.conf import settings

from .forms import EssayForm
from .models import Essay, EssayScore
from .ml_model.analyzer import score_car_essay

# Add this to your evaluator/ml_model/utils.py file or directly in views.py

def convert_to_json_serializable(obj):
    """Convert NumPy types to Python native types for JSON serialization."""
    import numpy as np
    
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.float32, np.float64)):  # Fixed this line
        return float(obj)
    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(i) for i in obj]
    else:
        return obj
def home(request):
    """Home page with essay submission form"""
    form = EssayForm()
    return render(request, 'evaluator/home.html', {'form': form})

@require_POST
def evaluate_essay(request):
    """Process essay submission and run evaluation"""
    form = EssayForm(request.POST)
    if form.is_valid():
        # Save the essay without committing to DB yet
        essay = form.save(commit=False)
        
        # Run the ML model evaluation
        try:
            # Score the essay using our ML model
            result = score_car_essay(essay.content)
            
            # Update essay with car topic information
            essay.is_car_topic = result.get('is_car_topic', False)
            essay.car_keyword_density = float(result.get('car_keyword_density', 0.0))
            essay.save()
            
            # Find this code in your evaluate_essay view function
            score = EssayScore(
                essay=essay,
                score=float(result.get('score', 0.0))
            )

            # Add normalization here
            # If the original score is out of 6, multiply by 10/6 to convert to a 10-point scale
            score.score = min(10.0, (score.score * 10.0) / 6.0)

            # Do the same for car-specific score
            if result.get('analysis') and result['analysis'].get('car_topic_analysis'):
                car_specific_score = result['analysis']['car_topic_analysis'].get('car_specific_score', 0.0)
                # Normalize to a 10-point scale
                score.car_specific_score = min(10.0, (float(car_specific_score) * 10.0) / 6.0)
            
            # Convert analysis data to JSON-serializable format
            if result.get('analysis'):
                # Use the utility function to convert NumPy types
                score.analysis_data = convert_to_json_serializable(result['analysis'])
            
            score.save()
            
            # Redirect to results page
            return redirect('evaluator:results', essay_id=essay.id)
        
        except Exception as e:
            # If there's an error in the ML process
            messages.error(request, f"Error evaluating essay: {str(e)}")
            return render(request, 'evaluator/home.html', {'form': form})
    else:
        # If form is not valid
        return render(request, 'evaluator/home.html', {'form': form})
def results(request, essay_id):
    """Show evaluation results for an essay"""
    essay = get_object_or_404(Essay, id=essay_id)
    
    # Check if the essay has a score
    try:
        score = essay.score
    except EssayScore.DoesNotExist:
        messages.error(request, "Essay evaluation not found.")
        return redirect('evaluator:home')
    
    context = {
        'essay': essay,
        'score': score,
    }
    
    return render(request, 'evaluator/results.html', context)

def history(request):
    """Show history of evaluated essays"""
    essays = Essay.objects.all().order_by('-created_at')[:20]  # Limit to recent 20
    return render(request, 'evaluator/history.html', {'essays': essays})

def about(request):
    """About page with information about the evaluation system"""
    return render(request, 'evaluator/about.html')