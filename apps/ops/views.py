"""
views.py (ops)
역할: 시스템 운영 지표를 집계해 JSON으로 반환하는 Metrics 엔드포인트.
      GET /v1/ops/metrics → throughput, failure_rate, latency(p50/p95/p99) 반환.
      Spring의 @Actuator /metrics 엔드포인트와 동일한 개념.
"""

import logging
from datetime import timedelta

import redis
import numpy as np
from django.db.models import F, ExpressionWrapper, DurationField
from django.utils import timezone
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework import status

logger = logging.getLogger(__name__)

from apps.jobs.models import InferenceJob, InferenceResult
from workers.redis_queue import REDIS_URL, DLQ_KEY

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
        # 측정 범위: InferenceJob.created_at (API 수신) → InferenceResult.created_at (결과 저장)
        # 즉, 큐 대기시간 + 배치 수집시간 + 추론시간을 모두 포함하는 end-to-end latency.
        # 순수 추론 시간(~277ms)과 다르며, 부하 상황에서는 큐 대기로 수 초까지 증가 가능.
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
            # end_to_end_latency: API 수신~결과 저장까지의 전체 소요 시간
            # 큐 대기 + 배치 수집 + 추론을 모두 포함 (순수 추론만이 아님)
            "end_to_end_latency_seconds": latency,
            "total_requests": total,
            "success_requests": success,
            "failed_requests": failed,
        })


class HealthView(APIView):
    """
    GET /v1/ops/health
    DB + Redis 연결 상태를 확인하는 헬스체크 엔드포인트.
    로드밸런서/컨테이너 오케스트레이터(K8s readinessProbe 등)가 인스턴스 상태 확인에 사용.
    의존 서비스 중 하나라도 응답 불가 시 503 반환 → 트래픽 라우팅 제외.
    """

    def get(self, request):
        # DB 연결 확인 — 간단한 쿼리로 MySQL 연결 가능 여부 테스트
        try:
            InferenceJob.objects.exists()
            db_ok = True
        except Exception:
            logger.exception("DB health check failed")  # 장애 시 스택 트레이스 기록
            db_ok = False

        # Redis 연결 확인 — PING/PONG으로 큐 브로커 가용성 테스트
        try:
            r = redis.from_url(REDIS_URL)
            r.ping()
            redis_ok = True
        except Exception:
            logger.exception("Redis health check failed")  # 장애 시 스택 트레이스 기록
            redis_ok = False

        overall = "ok" if (db_ok and redis_ok) else "degraded"
        http_status = status.HTTP_200_OK if overall == "ok" else status.HTTP_503_SERVICE_UNAVAILABLE

        return Response(
            {
                "status": overall,
                "db": "ok" if db_ok else "error",
                "redis": "ok" if redis_ok else "error",
            },
            status=http_status,
        )


class DLQView(APIView):
    """
    GET /v1/ops/dlq
    3회 재시도 후 최종 실패한 job 목록 조회.
    Redis dlq:failed_jobs 리스트에서 job_id를 읽어 DB 정보와 함께 반환.
    운영자가 장애 원인 파악 및 수동 재처리에 사용.
    """

    def get(self, request):
        r = redis.from_url(REDIS_URL, decode_responses=True)

        # DLQ 전체 조회 (0 ~ -1 = 처음부터 끝까지)
        job_ids = r.lrange(DLQ_KEY, 0, -1)

        if not job_ids:
            return Response({"count": 0, "jobs": []})

        # DB에서 해당 job들의 상세 정보 조회
        jobs = InferenceJob.objects.filter(pk__in=job_ids).values(
            "id", "status", "input_sha256", "created_at", "updated_at"
        )

        return Response({
            "count": len(job_ids),
            "jobs": list(jobs),
        })
