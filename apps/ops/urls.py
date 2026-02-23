"""
urls.py (ops)
역할: /v1/ops/* URL을 MetricsView에 연결.
"""

from django.urls import path
from .views import MetricsView

# config/urls.py에서 "v1/ops/" prefix가 이미 붙어있음
urlpatterns = [
    path("metrics", MetricsView.as_view(), name="ops-metrics"),  # GET /v1/ops/metrics
]
