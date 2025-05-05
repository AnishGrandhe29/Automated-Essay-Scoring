from django.urls import path
from . import views

app_name = 'evaluator'

urlpatterns = [
    path('', views.home, name='home'),
    path('evaluate/', views.evaluate_essay, name='evaluate_essay'),
    path('results/<int:essay_id>/', views.results, name='results'),
    path('history/', views.history, name='history'),
    path('about/', views.about, name='about'),
]