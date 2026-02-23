"""
test_views.py
역할: jobs API 엔드포인트의 비즈니스 로직과 에러 처리를 검증.
      Redis와 모델 추론은 Mock 처리 — 단위 테스트 범위만 검증.
      Spring의 @WebMvcTest + @MockBean 조합과 동일한 접근.
"""

import hashlib
import io
from unittest.mock import patch, MagicMock

import pytest
from django.urls import reverse

from apps.jobs.models import InferenceJob, InferenceResult


# ── POST /v1/jobs ───────────────────────────────────────────────

@pytest.mark.django_db
def test_create_job_success(api_client, sample_image_bytes, model_version):
    """정상 이미지 업로드 시 job이 생성되고 201을 반환하는지 검증."""
    # Redis 관련 함수 모두 Mock (실제 Redis 없이 테스트)
    # io.BytesIO에 .name 속성 추가 → DRF multipart가 파일로 인식
    image = io.BytesIO(sample_image_bytes)
    image.name = "test.png"

    with patch("apps.jobs.views.get_cache", return_value=None), \
         patch("apps.jobs.views.store_image"), \
         patch("apps.jobs.views.enqueue"), \
         patch("apps.jobs.views.set_cache"):

        response = api_client.post(
            "/v1/jobs",
            {"image": image},
            format="multipart",
        )

    assert response.status_code == 201
    assert response.data["status"] == "QUEUED"
    assert "id" in response.data

    # DB에 실제로 저장됐는지 확인
    assert InferenceJob.objects.filter(pk=response.data["id"]).exists()


@pytest.mark.django_db
def test_create_job_no_image(api_client, model_version):
    """image 필드 누락 시 400을 반환하는지 검증."""
    response = api_client.post("/v1/jobs", {}, format="multipart")
    assert response.status_code == 400
    assert "error" in response.data


@pytest.mark.django_db
def test_create_job_duplicate_returns_cached(api_client, sample_image_bytes, model_version):
    """
    동일 이미지 재요청 시 Redis 캐시에서 기존 job을 반환하는지 검증.
    (재추론 없이 200 반환)
    """
    # 기존에 처리된 job 생성
    existing_job = InferenceJob.objects.create(
        model=model_version,
        status=InferenceJob.Status.COMPLETED,
        input_sha256="abc123",
    )

    image = io.BytesIO(sample_image_bytes)
    image.name = "test.png"

    # get_cache가 기존 job_id를 반환하도록 Mock
    with patch("apps.jobs.views.get_cache", return_value=existing_job.id):
        response = api_client.post(
            "/v1/jobs",
            {"image": image},
            format="multipart",
        )

    # 새 job 생성 없이 기존 job 정보 반환
    assert response.status_code == 200
    assert response.data["id"] == existing_job.id


@pytest.mark.django_db
def test_create_job_no_model_version(api_client, sample_image_bytes):
    """등록된 ModelVersion이 없을 때 503을 반환하는지 검증."""
    image = io.BytesIO(sample_image_bytes)
    image.name = "test.png"

    with patch("apps.jobs.views.get_cache", return_value=None):
        response = api_client.post(
            "/v1/jobs",
            {"image": image},
            format="multipart",
        )

    assert response.status_code == 503


# ── GET /v1/jobs/{id} ───────────────────────────────────────────

@pytest.mark.django_db
def test_get_job_status(api_client, model_version):
    """존재하는 job 조회 시 200과 상태 정보를 반환하는지 검증."""
    job = InferenceJob.objects.create(
        model=model_version,
        status=InferenceJob.Status.IN_PROGRESS,
        input_sha256="deadbeef",
    )

    response = api_client.get(f"/v1/jobs/{job.id}")

    assert response.status_code == 200
    assert response.data["id"] == job.id
    assert response.data["status"] == "IN_PROGRESS"


@pytest.mark.django_db
def test_get_job_not_found(api_client):
    """존재하지 않는 job_id 조회 시 404를 반환하는지 검증."""
    response = api_client.get("/v1/jobs/99999")
    assert response.status_code == 404


# ── GET /v1/jobs/{id}/result ────────────────────────────────────

@pytest.mark.django_db
def test_get_result_completed(api_client, model_version):
    """COMPLETED job의 결과 조회 시 200과 결과 데이터를 반환하는지 검증."""
    job = InferenceJob.objects.create(
        model=model_version,
        status=InferenceJob.Status.COMPLETED,
        input_sha256="cafebabe",
    )
    InferenceResult.objects.create(
        job=job,
        output={"Pneumonia": 0.87, "Atelectasis": 0.12},
        top_label="Pneumonia",
    )

    response = api_client.get(f"/v1/jobs/{job.id}/result")

    assert response.status_code == 200
    assert response.data["top_label"] == "Pneumonia"
    assert "output" in response.data


@pytest.mark.django_db
def test_get_result_not_completed(api_client, model_version):
    """아직 QUEUED 상태인 job의 result 조회 시 409를 반환하는지 검증."""
    job = InferenceJob.objects.create(
        model=model_version,
        status=InferenceJob.Status.QUEUED,
        input_sha256="feedface",
    )

    response = api_client.get(f"/v1/jobs/{job.id}/result")

    assert response.status_code == 409
    assert "QUEUED" in response.data["error"]
