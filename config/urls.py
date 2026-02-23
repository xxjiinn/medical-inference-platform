from django.urls import path, include

# 최상위 URL 라우팅 (Spring의 @RequestMapping 분기와 동일)
# 각 앱의 urls.py로 위임 (include)
urlpatterns = [
    path("v1/", include("apps.jobs.urls")),      # /v1/jobs/* → apps/jobs/urls.py
    path("v1/ops/", include("apps.ops.urls")),   # /v1/ops/* → apps/ops/urls.py
]
