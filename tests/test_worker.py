"""
test_worker.py
역할: workers/worker.py의 핵심 추론 로직(process_batch, _handle_failed_jobs)과
      workers/main.py의 _recover_stuck_jobs()를 Mock 객체로 단위 테스트.
      실제 모델·Redis 없이 DB 상태 변화만 검증.
      Spring의 @MockBean + @SpringBootTest(webEnvironment=NONE)과 동일한 개념.
"""

from unittest.mock import patch, MagicMock
import pytest

from apps.jobs.models import InferenceJob, InferenceResult


# ── 공통 픽스처 ─────────────────────────────────────────────────

@pytest.fixture
def inference_job(db, model_version):
    """
    테스트용 InferenceJob 레코드 생성 (status=QUEUED, sha256 고정).
    model_version은 conftest.py의 ModelVersion 픽스처 의존.
    """
    return InferenceJob.objects.create(
        model=model_version,
        input_sha256="abc123deadbeef",
    )


@pytest.fixture
def mock_loader():
    """
    Mock 추론 로더: 실제 DenseNet121 로드 없이 preprocess/predict_batch를 시뮬레이션.
    Spring의 @MockBean ModelService와 동일 — 인터페이스만 맞추면 됨.
    """
    loader = MagicMock()
    # preprocess: bytes 입력 → 더미 텐서 반환
    loader.preprocess.return_value = MagicMock()
    # predict_batch: 배치 추론 결과 — Effusion 점수가 가장 높은 딕셔너리
    loader.predict_batch.return_value = [{"Effusion": 0.9, "Pneumonia": 0.1}]
    return loader


# ── process_batch() 테스트 ──────────────────────────────────────

@pytest.mark.django_db
def test_process_batch_success(inference_job, mock_loader):
    """
    정상 경로: 이미지 조회·전처리·배치 추론 모두 성공 시
    InferenceResult 생성 + Job status=COMPLETED 변경을 검증.
    """
    from workers.worker import process_batch

    with (
        # Redis 이미지 조회 mock: 항상 더미 bytes 반환
        patch("workers.worker.fetch_image_bytes", return_value=b"fake_image"),
        # 실제 모델 로드 없이 mock 로더로 교체 (HuggingFace 다운로드 방지)
        patch("workers.worker.get_loader", return_value=mock_loader),
        # SIGALRM 타임아웃 타이머 비활성화 (테스트 프로세스 보호)
        patch("signal.alarm"),
    ):
        process_batch([inference_job.id])

    # DB에서 최신 상태 다시 로드 (메모리 캐시 무효화)
    inference_job.refresh_from_db()

    # Job 상태 검증: QUEUED → IN_PROGRESS → COMPLETED
    assert inference_job.status == InferenceJob.Status.COMPLETED

    # InferenceResult 생성 여부 검증
    result = InferenceResult.objects.get(job=inference_job)
    # top_label: predict_batch 결과 중 가장 높은 점수(Effusion=0.9)
    assert result.top_label == "Effusion"


@pytest.mark.django_db
def test_process_batch_image_not_found(inference_job, mock_loader):
    """
    이미지 없음 경로: Redis에 이미지가 TTL 만료·없을 때 (None 반환)
    Job이 큐에 재등록되고 InferenceResult는 생성되지 않는지 검증.
    """
    from workers.worker import process_batch

    mock_redis = MagicMock()
    mock_redis.incr.return_value = 1   # 첫 번째 재시도 (1 <= MAX_RETRIES=3)
    mock_redis.expire.return_value = 1  # TTL 설정 성공

    with (
        patch("workers.worker.fetch_image_bytes", return_value=None),  # 이미지 없음
        patch("workers.worker.get_loader", return_value=mock_loader),
        # _handle_failed_jobs 내 retry 카운터용 Redis 연결 mock
        patch("workers.worker.redis.from_url", return_value=mock_redis),
        # enqueue() 내부의 get_redis() mock (큐 재등록 경로)
        patch("workers.redis_queue.get_redis", return_value=mock_redis),
        patch("signal.alarm"),
    ):
        process_batch([inference_job.id])

    # 추론 실패 → InferenceResult 없음
    assert not InferenceResult.objects.filter(job=inference_job).exists()

    # LPUSH로 inference:queue에 재등록됐는지 검증
    mock_redis.lpush.assert_called_once_with("inference:queue", str(inference_job.id))


@pytest.mark.django_db
def test_process_batch_preprocess_failure(inference_job, mock_loader):
    """
    전처리 실패 경로: preprocess()가 예외를 던질 때
    InferenceResult 없음 + 재시도 큐 등록을 검증.
    """
    from workers.worker import process_batch

    # preprocess가 ValueError를 던지도록 설정 (손상된 이미지 시뮬레이션)
    mock_loader.preprocess.side_effect = ValueError("Invalid image format")

    mock_redis = MagicMock()
    mock_redis.incr.return_value = 1  # 첫 번째 재시도

    with (
        patch("workers.worker.fetch_image_bytes", return_value=b"bad_image"),
        patch("workers.worker.get_loader", return_value=mock_loader),
        patch("workers.worker.redis.from_url", return_value=mock_redis),
        patch("workers.redis_queue.get_redis", return_value=mock_redis),
        patch("signal.alarm"),
    ):
        process_batch([inference_job.id])

    # 전처리 실패 → 추론 미실행 → InferenceResult 없음
    assert not InferenceResult.objects.filter(job=inference_job).exists()


@pytest.mark.django_db
def test_process_batch_nonexistent_job(mock_loader):
    """
    DB에 없는 job_id: 조용히 스킵하고 예외 없이 종료되는지 검증.
    삭제된 Job이 큐에 남아있는 엣지 케이스 처리 확인.
    """
    from workers.worker import process_batch

    with (
        patch("workers.worker.fetch_image_bytes", return_value=b"fake"),
        patch("workers.worker.get_loader", return_value=mock_loader),
        patch("signal.alarm"),
    ):
        # 존재하지 않는 job_id — 예외 없이 조용히 스킵되어야 함
        process_batch([99999])


# ── _handle_failed_jobs() 테스트 ────────────────────────────────

@pytest.mark.django_db
def test_handle_failed_jobs_retry(inference_job):
    """
    재시도 경로: attempt <= MAX_RETRIES이면
    Job이 FAILED로 확정되지 않고 inference:queue에 재등록되는지 검증.
    """
    from workers.worker import _handle_failed_jobs

    mock_redis = MagicMock()
    mock_redis.incr.return_value = 1  # 첫 번째 재시도 (1 <= MAX_RETRIES=3)

    with (
        # retry 카운터(INCR/EXPIRE) 처리용 Redis mock
        patch("workers.worker.redis.from_url", return_value=mock_redis),
        # enqueue() 내부 LPUSH용 get_redis() mock
        patch("workers.redis_queue.get_redis", return_value=mock_redis),
    ):
        _handle_failed_jobs([inference_job])

    inference_job.refresh_from_db()

    # 재시도이므로 FAILED로 확정되지 않음
    assert inference_job.status != InferenceJob.Status.FAILED

    # inference:queue에 LPUSH 호출 확인 (재시도 등록)
    mock_redis.lpush.assert_called_with("inference:queue", str(inference_job.id))


@pytest.mark.django_db
def test_handle_failed_jobs_dlq(inference_job):
    """
    DLQ 경로: attempt > MAX_RETRIES이면
    Job이 FAILED로 확정되고 dlq:failed_jobs에 job_id가 push되는지 검증.
    Dead Letter Queue 동작 확인.
    """
    from workers.worker import _handle_failed_jobs
    from django.conf import settings

    mock_redis = MagicMock()
    # MAX_RETRIES+1 → 재시도 횟수 소진 (DLQ 경로)
    mock_redis.incr.return_value = settings.MAX_RETRIES + 1

    with (
        patch("workers.worker.redis.from_url", return_value=mock_redis),
        patch("workers.redis_queue.get_redis", return_value=mock_redis),
    ):
        _handle_failed_jobs([inference_job])

    inference_job.refresh_from_db()

    # Job 상태 FAILED 확정 검증
    assert inference_job.status == InferenceJob.Status.FAILED

    # DLQ(dlq:failed_jobs)에 job_id push 검증
    mock_redis.lpush.assert_called_with("dlq:failed_jobs", inference_job.id)

    # retry 카운터 키 삭제 검증 (Redis 정리)
    mock_redis.delete.assert_called_once_with(f"retry:{inference_job.id}")


# ── _recover_stuck_jobs() 테스트 ─────────────────────────────────

@pytest.mark.django_db
def test_recover_stuck_jobs_requeue(inference_job):
    """
    stuck job 복구 경로: attempt <= MAX_RETRIES이면
    status가 QUEUED로 돌아오고 inference:queue에 재등록되는지 검증.
    """
    from workers.main import _recover_stuck_jobs
    from django.conf import settings

    # updated_at을 20분 전으로 강제 설정해 stuck 조건 충족
    InferenceJob.objects.filter(pk=inference_job.id).update(
        status=InferenceJob.Status.IN_PROGRESS,
    )
    from django.utils import timezone
    from datetime import timedelta
    InferenceJob.objects.filter(pk=inference_job.id).update(
        updated_at=timezone.now() - timedelta(minutes=20),
    )

    mock_redis = MagicMock()
    mock_redis.incr.return_value = 1  # 첫 번째 recovery 시도 (1 <= MAX_RETRIES=3)

    with (
        patch("workers.main.redis.from_url", return_value=mock_redis),
        patch("workers.redis_queue.get_redis", return_value=mock_redis),
    ):
        _recover_stuck_jobs()

    inference_job.refresh_from_db()
    # recovery 후 QUEUED로 복귀 검증
    assert inference_job.status == InferenceJob.Status.QUEUED
    # inference:queue에 재등록 검증
    mock_redis.lpush.assert_called_with("inference:queue", str(inference_job.id))


@pytest.mark.django_db
def test_recover_stuck_jobs_dlq_on_max_retries(inference_job):
    """
    stuck job 복구 DLQ 경로: recovery 시도가 MAX_RETRIES 초과 시
    FAILED 확정 + DLQ push되는지 검증.
    mid-inference SIGKILL 무한루프 방지 메커니즘 확인.
    """
    from workers.main import _recover_stuck_jobs
    from django.conf import settings

    InferenceJob.objects.filter(pk=inference_job.id).update(
        status=InferenceJob.Status.IN_PROGRESS,
    )
    from django.utils import timezone
    from datetime import timedelta
    InferenceJob.objects.filter(pk=inference_job.id).update(
        updated_at=timezone.now() - timedelta(minutes=20),
    )

    mock_redis = MagicMock()
    # MAX_RETRIES+1 → 재시도 횟수 소진 (DLQ 경로)
    mock_redis.incr.return_value = settings.MAX_RETRIES + 1

    with (
        patch("workers.main.redis.from_url", return_value=mock_redis),
        patch("workers.redis_queue.get_redis", return_value=mock_redis),
    ):
        _recover_stuck_jobs()

    inference_job.refresh_from_db()
    # recovery 횟수 초과 → FAILED 확정
    assert inference_job.status == InferenceJob.Status.FAILED
    # DLQ push 검증
    mock_redis.lpush.assert_called_with("dlq:failed_jobs", inference_job.id)
    # retry 카운터 삭제 검증
    mock_redis.delete.assert_called_once_with(f"retry:{inference_job.id}")
