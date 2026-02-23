"""
views.py (ops)
역할: 시스템 운영 지표를 집계해 JSON으로 반환하는 Metrics 엔드포인트.
      GET /v1/ops/metrics → throughput, failure_rate, latency(p50/p95/p99) 반환.
      Spring의 @Actuator /metrics 엔드포인트와 동일한 개념.
"""

from datetime import timedelta

import numpy as np
from django.db.models import F, ExpressionWrapper, DurationField
from django.utils import timezone
from rest_framework.response import Response
from rest_framework.views import APIView

from apps.jobs.models import InferenceJob, InferenceResult

# 지표 집계 시간 윈도우 (최근 5분)
METRICS_WINDOW_MINUTES = 5


class MetricsView(APIView):
    """
    GET /v1/ops/metrics
    최근 5분간의 추론 처리량, 실패율, 레이턴시 백분위수 반환.
    """

    def get(self, request):
        # 5분 전 시각 기준점
        since = timezone.now() - timedelta(minutes=METRICS_WINDOW_MINUTES)

        # ── 요청 수 집계 ──────────────────────────────────────────
        # 최근 5분간 생성된 전체 job 수
        total = InferenceJob.objects.filter(created_at__gte=since).count()
        # 그 중 성공한 job 수
        success = InferenceJob.objects.filter(
            created_at__gte=since,
            status=InferenceJob.Status.COMPLETED,
        ).count()
        # 그 중 실패한 job 수
        failed = InferenceJob.objects.filter(
            created_at__gte=since,
            status=InferenceJob.Status.FAILED,
        ).count()

        # ── 처리량 (Throughput) ───────────────────────────────────
        # RPS = 성공한 job 수 / 윈도우(초)
        # (COMPLETED 기준 — 실제로 결과를 만들어낸 요청만 집계)
        window_seconds = METRICS_WINDOW_MINUTES * 60
        throughput = round(success / window_seconds, 3)

        # ── 실패율 (Failure Rate) ─────────────────────────────────
        # total이 0이면 division by zero 방지
        failure_rate = round(failed / total, 4) if total > 0 else 0.0

        # ── 레이턴시 계산 ─────────────────────────────────────────
        # InferenceResult.created_at - InferenceJob.created_at = 추론 소요 시간
        # annotate(): SQL에서 컬럼 간 연산 결과를 새 필드로 추가
        # ExpressionWrapper: Django ORM에서 duration 타입 연산을 명시적으로 감쌈
        # F('job__created_at'): InferenceResult → InferenceJob FK 역참조
        latency_qs = (
            InferenceResult.objects
            .filter(job__created_at__gte=since)
            .annotate(
                duration=ExpressionWrapper(
                    F("created_at") - F("job__created_at"),
                    output_field=DurationField(),
                )
            )
            .values_list("duration", flat=True)
        )

        # timedelta 리스트 → float(초) 리스트로 변환
        durations_sec = [d.total_seconds() for d in latency_qs if d is not None]

        if durations_sec:
            arr = np.array(durations_sec)
            latency = {
                "p50": round(float(np.percentile(arr, 50)), 3),
                "p95": round(float(np.percentile(arr, 95)), 3),
                "p99": round(float(np.percentile(arr, 99)), 3),
            }
        else:
            # 데이터 없을 때 null 대신 0으로 반환 (클라이언트 파싱 편의)
            latency = {"p50": 0.0, "p95": 0.0, "p99": 0.0}

        return Response({
            "window_minutes": METRICS_WINDOW_MINUTES,  # 집계 기준 시간 윈도우
            "throughput_rps": throughput,               # 초당 성공 요청 수
            "failure_rate": failure_rate,               # 실패율 (0.0 ~ 1.0)
            "latency_seconds": latency,                 # 추론 소요 시간 백분위수
            "total_requests": total,
            "success_requests": success,
            "failed_requests": failed,
        })
