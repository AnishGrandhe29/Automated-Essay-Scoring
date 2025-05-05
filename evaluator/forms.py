from django import forms
from .models import Essay

class EssayForm(forms.ModelForm):
    class Meta:
        model = Essay
        fields = ['title', 'content']
        widgets = {
            'title': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'Enter essay title',
            }),
            'content': forms.Textarea(attrs={
                'class': 'form-control',
                'placeholder': 'Enter your essay about cars here...',
                'rows': 15,
            }),
        }
        
    def clean_content(self):
        content = self.cleaned_data.get('content')
        if len(content.split()) < 50:
            raise forms.ValidationError("Your essay is too short. Please write at least 50 words.")
        return content