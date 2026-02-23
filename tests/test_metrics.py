"""
test_metrics.py
역할: GET /v1/ops/metrics 응답 구조와 집계 로직을 검증.
"""

import pytest
from datetime import timedelta
from django.utils import timezone

from apps.jobs.models import InferenceJob, InferenceResult


@pytest.mark.django_db
def test_metrics_returns_correct_structure(api_client):
    """데이터 없을 때도 올바른 구조의 JSON을 반환하는지 검증."""
    response = api_client.get("/v1/ops/metrics")

    assert response.status_code == 200
    # 필수 키 존재 여부 확인
    for key in ["throughput_rps", "failure_rate", "latency_seconds",
                "total_requests", "success_requests", "failed_requests"]:
        assert key in response.data, f"Missing key: {key}"

    # 데이터 없을 때 latency 0으로 반환하는지 확인
    assert response.data["latency_seconds"]["p50"] == 0.0
    assert response.data["total_requests"] == 0


@pytest.mark.django_db
def test_metrics_counts_correctly(api_client, model_version):
    """최근 5분간 COMPLETED/FAILED job 수가 정확히 집계되는지 검증."""
    now = timezone.now()

    # 최근 5분 내 데이터 생성
    # auto_now_add=True라 create()에서 created_at을 지정할 수 없음.
    # update()는 auto_now_add를 우회해 직접 값 설정 가능 (Spring의 @Modifying 쿼리와 동일)
    for _ in range(3):
        job = InferenceJob.objects.create(
            model=model_version,
            status=InferenceJob.Status.COMPLETED,
            input_sha256=f"hash_ok_{_}",
        )
        InferenceJob.objects.filter(pk=job.pk).update(
            created_at=now - timedelta(minutes=2)  # 2분 전 → 집계 대상
        )
    job_fail = InferenceJob.objects.create(
        model=model_version,
        status=InferenceJob.Status.FAILED,
        input_sha256="hash_fail",
    )
    InferenceJob.objects.filter(pk=job_fail.pk).update(
        created_at=now - timedelta(minutes=1)
    )
    # 5분 밖 데이터 (집계 제외 대상)
    job_old = InferenceJob.objects.create(
        model=model_version,
        status=InferenceJob.Status.COMPLETED,
        input_sha256="hash_old",
    )
    InferenceJob.objects.filter(pk=job_old.pk).update(
        created_at=now - timedelta(minutes=10)  # 10분 전 → 집계 제외
    )

    response = api_client.get("/v1/ops/metrics")

    assert response.status_code == 200
    assert response.data["total_requests"] == 4      # 최근 5분: 3 completed + 1 failed
    assert response.data["success_requests"] == 3
    assert response.data["failed_requests"] == 1
    # failure_rate = 1 / 4
    assert abs(response.data["failure_rate"] - 0.25) < 0.001


@pytest.mark.django_db
def test_metrics_latency_calculated(api_client, model_version):
    """InferenceResult가 있을 때 latency가 0이 아닌 값으로 계산되는지 검증."""
    now = timezone.now()

    job = InferenceJob.objects.create(
        model=model_version,
        status=InferenceJob.Status.COMPLETED,
        input_sha256="hash_latency",
    )
    # result.created_at이 job.created_at보다 2초 늦도록 설정
    # auto_now_add=True 우회: create() 후 update()로 직접 설정
    result = InferenceResult.objects.create(
        job=job,
        output={"Pneumonia": 0.9},
        top_label="Pneumonia",
    )
    InferenceResult.objects.filter(pk=result.pk).update(
        created_at=job.created_at + timedelta(seconds=2)
    )

    response = api_client.get("/v1/ops/metrics")

    # latency p50이 0보다 크면 계산된 것
    assert response.data["latency_seconds"]["p50"] > 0
