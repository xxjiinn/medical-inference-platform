"""
worker.py
역할: Redis 큐에서 job 배치를 수집해 한 번의 forward pass로 추론하고 결과를 DB에 저장.
      Phase 6 Micro-batching: 30ms 윈도우 내 job들을 묶어 배치 추론으로 처리 효율 향상.
      Phase 7: INFERENCE_ENGINE 환경변수로 PyTorch/ONNX 엔진 전환 지원.
      실패 job은 개별 재시도(MAX_RETRIES) 후 FAILED 처리.
"""

import os
import sys
import signal
import logging
import redis as redis_lib

# 프로젝트 루트를 Python 경로에 추가 (독립 프로세스로 실행되므로 필요)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Django ORM 사용을 위한 초기화 — 반드시 모델 import 전에 실행해야 함
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
import django
django.setup()

from django.conf import settings
from apps.jobs.models import InferenceJob, InferenceResult
from workers.queue import collect_batch, REDIS_URL

# INFERENCE_ENGINE 설정에 따라 로더 선택
# "onnx"면 OnnxLoader, 그 외(기본값 "pytorch")면 ModelLoader 사용
if settings.INFERENCE_ENGINE == "onnx":
    from workers.onnx_loader import get_onnx_loader as get_loader
else:
    from workers.model_loader import get_loader

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[Worker %(process)d] %(message)s")


# ── 이미지 가져오기 ────────────────────────────────────────────
def fetch_image_bytes(sha256: str) -> bytes | None:
    """Redis에서 이미지 bytes 조회. 만료되었거나 없으면 None 반환."""
    r = redis_lib.from_url(REDIS_URL, decode_responses=False)
    return r.get(f"image:{sha256}")


# ── 타임아웃 처리 ──────────────────────────────────────────────
def _timeout_handler(signum, frame):
    """SIGALRM 수신 시 — 배치 추론 전체가 제한 시간 초과."""
    raise TimeoutError("Batch inference timed out")

signal.signal(signal.SIGALRM, _timeout_handler)


# ── 배치 처리 핵심 함수 ────────────────────────────────────────
def process_batch(job_ids: list[int]) -> None:
    """
    여러 job을 한 번의 배치 추론으로 처리.

    단계:
      1. DB에서 Job 조회 + 상태 IN_PROGRESS로 일괄 업데이트
      2. Redis에서 이미지 bytes 가져와 전처리 (bytes -> tensor)
      3. 유효한 텐서만 배치로 묶어 단일 forward pass 실행
      4. 결과를 각 job의 InferenceResult로 저장 + COMPLETED 처리
      5. 이미지 없거나 전처리 실패한 job -> 재시도 또는 FAILED 처리
    """
    loader = get_loader()

    # 1. DB에서 Job 객체 일괄 조회
    jobs = {
        job.id: job
        for job in InferenceJob.objects.filter(pk__in=job_ids)
    }

    # DB에 없는 job_id 경고 (이미 삭제된 경우 등)
    for jid in job_ids:
        if jid not in jobs:
            logger.warning(f"Job {jid} not found in DB, skipping.")

    if not jobs:
        return

    # 2. 모든 Job 상태를 IN_PROGRESS로 일괄 업데이트
    #    update() = SQL UPDATE ... WHERE id IN (...) — 루프 없이 한 쿼리로 처리
    InferenceJob.objects.filter(pk__in=list(jobs.keys())).update(
        status=InferenceJob.Status.IN_PROGRESS
    )
    logger.info(f"Batch of {len(jobs)} jobs -> IN_PROGRESS")

    # 3. 각 Job의 이미지 bytes를 Redis에서 가져와 전처리
    valid_jobs = []    # 정상적으로 전처리된 (job, tensor) 쌍
    failed_jobs = []   # 이미지 없거나 전처리 실패한 job

    for job in jobs.values():
        image_bytes = fetch_image_bytes(job.input_sha256)
        if image_bytes is None:
            logger.warning(f"Job {job.id}: image not found in Redis (expired?)")
            failed_jobs.append(job)
            continue
        try:
            tensor = loader.preprocess(image_bytes)  # bytes -> (1,1,224,224) tensor
            valid_jobs.append((job, tensor))
        except Exception as e:
            logger.warning(f"Job {job.id}: preprocess failed — {e}")
            failed_jobs.append(job)

    # 4. 유효한 job들을 배치 추론
    if valid_jobs:
        tensors = [t for _, t in valid_jobs]  # tensor 리스트만 추출

        # 배치 추론 타임아웃 = 단일 타임아웃 × 배치 크기
        # (배치가 클수록 시간이 오래 걸리므로 비례하여 여유를 줌)
        signal.alarm(settings.INFERENCE_TIMEOUT * len(tensors))
        try:
            # 핵심: N개 텐서를 (N,1,224,224)로 묶어 한 번의 forward pass 실행
            batch_scores = loader.predict_batch(tensors)
        except TimeoutError:
            # 배치 전체 타임아웃 -> 전부 개별 재시도 대상으로 이동
            logger.error(f"Batch inference timed out. Moving all to retry.")
            failed_jobs.extend(job for job, _ in valid_jobs)
            valid_jobs = []
            batch_scores = []
        except Exception as e:
            logger.error(f"Batch inference error: {e}. Moving all to retry.")
            failed_jobs.extend(job for job, _ in valid_jobs)
            valid_jobs = []
            batch_scores = []
        finally:
            signal.alarm(0)  # 타이머 해제

        # 5. 배치 추론 성공한 job들 결과 저장
        for (job, _), scores in zip(valid_jobs, batch_scores):
            top_label = max(scores, key=lambda k: scores[k])
            InferenceResult.objects.create(job=job, output=scores, top_label=top_label)
            job.status = InferenceJob.Status.COMPLETED
            job.save(update_fields=["status", "updated_at"])
            logger.info(f"Job {job.id} COMPLETED — top_label={top_label}")

    # 6. 실패 job들: 재시도 횟수 체크 후 처리
    #    재시도 횟수는 Redis에 카운터로 관리 (DB 추가 컬럼 없이)
    if failed_jobs:
        _handle_failed_jobs(failed_jobs)


def _handle_failed_jobs(jobs: list) -> None:
    """
    실패한 job들의 재시도 횟수를 Redis 카운터로 추적.
    MAX_RETRIES 미만이면 큐에 재등록, 초과 시 FAILED 처리.

    Redis 재시도 카운터 키: retry:{job_id}  (TTL 1시간)
    """
    from workers.queue import enqueue
    r = redis_lib.from_url(REDIS_URL, decode_responses=True)

    for job in jobs:
        retry_key = f"retry:{job.id}"
        # INCR: 카운터 없으면 0에서 시작해 1 증가, 있으면 +1
        attempt = r.incr(retry_key)
        r.expire(retry_key, 3600)  # 1시간 후 자동 삭제

        if attempt <= settings.MAX_RETRIES:
            # 재시도 가능 -> 큐 맨 뒤에 다시 등록
            logger.warning(f"Job {job.id}: retry {attempt}/{settings.MAX_RETRIES}, re-enqueuing.")
            enqueue(job.id)
        else:
            # 재시도 횟수 소진 -> FAILED 확정
            job.status = InferenceJob.Status.FAILED
            job.save(update_fields=["status", "updated_at"])
            r.delete(retry_key)  # 카운터 정리
            logger.error(f"Job {job.id}: FAILED after {settings.MAX_RETRIES} retries.")


# ── 워커 메인 루프 ─────────────────────────────────────────────
def run_worker():
    """
    워커 프로세스의 메인 루프.
    모델 로드 -> Redis 큐 배치 폴링 -> 배치 추론 반복.
    SIGTERM 수신 시 현재 배치 완료 후 종료 (Graceful Shutdown).
    """
    shutdown = False

    def handle_sigterm(signum, frame):
        nonlocal shutdown
        logger.info("Received SIGTERM. Finishing current batch and shutting down...")
        shutdown = True

    signal.signal(signal.SIGTERM, handle_sigterm)

    # 모델 로드 (HuggingFace 캐시 또는 다운로드 후 메모리에 올림)
    loader = get_loader()
    loader.load()
    logger.info("Model loaded. Worker ready.")

    while not shutdown:
        # 30ms 윈도우로 배치 수집 (최대 8개)
        # 큐가 비면 BRPOP이 5초 대기 후 빈 리스트 반환
        job_ids = collect_batch(
            max_wait_ms=settings.BATCH_WINDOW_MS,
            max_size=8,
        )

        if not job_ids:
            # 큐가 비어있음 — shutdown 여부 체크 후 다시 대기
            continue

        logger.info(f"Collected batch: {job_ids}")
        process_batch(job_ids)

    logger.info("Worker shut down cleanly.")


if __name__ == "__main__":
    run_worker()
