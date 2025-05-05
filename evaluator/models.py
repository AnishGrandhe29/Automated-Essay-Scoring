from django.db import models
import json

class Essay(models.Model):
    title = models.CharField(max_length=255)
    content = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    is_car_topic = models.BooleanField(default=False)
    car_keyword_density = models.FloatField(default=0.0)
    
    def __str__(self):
        return self.title
    
    class Meta:
        verbose_name_plural = "Essays"
        ordering = ['-created_at']

class EssayScore(models.Model):
    essay = models.OneToOneField(Essay, on_delete=models.CASCADE, related_name='score')
    score = models.FloatField(default=0.0)
    car_specific_score = models.FloatField(default=0.0)
    analysis_data = models.JSONField(default=dict)
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"Score for {self.essay.title}: {self.score}"
    
    def get_sentence_analysis(self):
        return self.analysis_data.get('sentence_analysis', {})
    
    def get_vocabulary_analysis(self):
        return self.analysis_data.get('vocabulary_analysis', {})
    
    def get_car_topic_analysis(self):
        return self.analysis_data.get('car_topic_analysis', {})
    
    def get_recommendations(self):
        car_analysis = self.get_car_topic_analysis()
        if car_analysis:
            return car_analysis.get('recommendations', [])
        return []
    
    def get_strongest_category(self):
        car_analysis = self.get_car_topic_analysis()
        if car_analysis:
            return car_analysis.get('strongest_category', '')
        return ''
    
    def get_areas_for_improvement(self):
        car_analysis = self.get_car_topic_analysis()
        if car_analysis:
            return car_analysis.get('areas_for_improvement', [])
        return []
    
    class Meta:
        verbose_name_plural = "Essay Scores"
        ordering = ['-created_at']