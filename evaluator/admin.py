from django.contrib import admin
from .models import Essay, EssayScore

@admin.register(Essay)
class EssayAdmin(admin.ModelAdmin):
    list_display = ('id', 'title', 'created_at', 'is_car_topic')
    list_filter = ('is_car_topic', 'created_at')
    search_fields = ('title', 'content')

@admin.register(EssayScore)
class EssayScoreAdmin(admin.ModelAdmin):
    list_display = ('essay', 'score', 'car_specific_score', 'created_at')
    list_filter = ('created_at',)
    search_fields = ('essay__title',)