"""
ASGI config for car_essay_evaluator project.
"""

import os

from django.core.asgi import get_asgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'car_essay_evaluator.settings')

application = get_asgi_application()
