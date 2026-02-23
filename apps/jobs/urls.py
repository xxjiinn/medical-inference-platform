"""
urls.py
역할: URL 패턴과 View를 연결.
      Spring의 @RequestMapping / @GetMapping / @PostMapping 선언과 동일.
"""

from django.urls import path
from .views import JobCreateView, JobStatusView, JobResultView

# config/urls.py에서 "v1/" prefix가 이미 붙어있음
# 따라서 여기서는 "jobs" 이하 경로만 정의
urlpatterns = [
    path("jobs", JobCreateView.as_view(), name="job-create"),           # POST /v1/jobs
    path("jobs/<int:job_id>", JobStatusView.as_view(), name="job-status"),  # GET /v1/jobs/{id}
    path("jobs/<int:job_id>/result", JobResultView.as_view(), name="job-result"),  # GET /v1/jobs/{id}/result
]
