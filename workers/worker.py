"""
worker.py
역할: Redis 큐에서 job을 꺼내 배치 추론 후 결과를 DB에 저장.

핵심 설계:
  - 30ms 배치 윈도우(micro-batching): 첫 job을 BRPOP으로 기다린 뒤
    30ms 동안 추가 job을 모아 한 번의 forward pass로 처리한다.
    CPU에서는 forward pass가 선형 증가해 throughput 이득은 없다.
    실질 효과는 N개 job 상태 전환을 단일 쿼리로 처리하는 DB 오버헤드 절감.
    GPU로 전환 시 배치 병렬화로 throughput과 latency 모두 개선된다.
  - INFERENCE_ENGINE 환경변수로 PyTorch/ONNX 엔진 전환 지원
    (ONNX는 현재 모델 구조상 변환 불가 — ADR-003 참고).
  - 실패 job은 Redis 재시도 카운터로 추적해 MAX_RETRIES 초과 시 FAILED + DLQ 처리.
"""

import os
import sys
import time
import json
import signal
import logging
import redis
from django.db import transaction

# 프로젝트 루트를 Python 경로에 추가 (독립 프로세스로 실행되므로 필요)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Django ORM 사용을 위한 초기화 — 반드시 모델 import 전에 실행해야 함
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
import django
django.setup()

from django.conf import settings
from apps.jobs.models import InferenceJob, InferenceResult
from workers.redis_queue import collect_batch, enqueue, REDIS_URL, DLQ_KEY

# INFERENCE_ENGINE 설정에 따라 로더 선택
# "onnx"면 OnnxLoader, 그 외(기본값 "pytorch")면 ModelLoader 사용
if settings.INFERENCE_ENGINE == "onnx":
    from workers.onnx_loader import get_onnx_loader as get_loader
else:
    from workers.model_loader import get_loader

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[Worker %(process)d] %(message)s")


def log(event: str, **kwargs):
    """구조화 로그 출력. job_id 등 컨텍스트를 JSON으로 기록해 추적 가능하게 함."""
    logger.info(json.dumps({"event": event, **kwargs}, ensure_ascii=False))


# ── 이미지 가져오기 ────────────────────────────────────────────
def fetch_image_bytes(sha256: str) -> bytes | None:
    """Worker 프로세스가 image:{sha256} 형태의 키를 생성하고 Redis에서 이미지 bytes 조회. 만료되었거나 없으면 None 반환."""
    r = redis.from_url(REDIS_URL, decode_responses=False)
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
      (job_id를 DB에서 조회 -> input_sha256 가져옴. -> redis에서 그 input_sha256으로 이미지 bytes 추출)
      2. Redis에서 이미지 bytes 가져와 전처리 (bytes -> tensor)
      3. 유효한 텐서만 배치로 묶어 단일 forward pass 실행
      (즉, 여러 갸의 텐서를 하나의 (N, 1, 224, 224) 형태로 만들어서 한번에 모델에게 보내는 것)
      4. 결과((N, 18)형태의 행렬)를 각 job의 InferenceResult로 저장 + COMPLETED 처리
      5. 이미지 없거나 전처리 실패한 job -> 재시도 또는 FAILED 처리
    """
    loader = get_loader()

    # 1. DB에서 Job 객체 조회 + IN_PROGRESS로 원자적 전환
    #    select_for_update(skip_locked=True): 이미 다른 워커가 처리 중인 row는 건너뜀.
    #    WORKER_COUNT > 1 환경에서 같은 job이 두 워커에게 동시 처리되는 것을 DB 락으로 방지.
    #    transaction.atomic() 블록 안에서 락을 잡고 즉시 IN_PROGRESS로 갱신 후 해제.
    with transaction.atomic():
        locked = list(
            InferenceJob.objects.select_for_update(skip_locked=True).filter(
                pk__in=job_ids,
                status=InferenceJob.Status.QUEUED,  # 이미 IN_PROGRESS로 선점된 job 제외
            )
        )
        if not locked:
            # 모든 job이 다른 워커에게 선점됨 (WORKER_COUNT > 1 동시 처리 상황)
            logger.warning(f"⚠️ job_ids={job_ids} 전부 다른 워커에게 선점됨 — 스킵")
            return

        jobs = {job.id: job for job in locked}

        # DB에 없는 job_id 경고
        for jid in job_ids:
            if jid not in jobs:
                logger.warning(f"⚠️ Job {jid} DB에 없거나 이미 선점됨, 스킵")

        # 락 보유 중 IN_PROGRESS로 일괄 갱신 — transaction 커밋 시 락 해제
        InferenceJob.objects.filter(pk__in=list(jobs.keys())).update(
            status=InferenceJob.Status.IN_PROGRESS
        )
    batch_start = time.time()  # 배치 전체 처리 시작 시각 기록
    log("batch_start", job_ids=list(jobs.keys()), batch_size=len(jobs))

    # 3. 각 Job의 이미지 bytes를 Redis에서 가져와 전처리
    valid_jobs = []    # 정상적으로 전처리된 (job, tensor) 쌍
    failed_jobs = []   # 이미지 없거나 전처리 실패한 job

    for job in jobs.values():
        image_bytes = fetch_image_bytes(job.input_sha256)
        if image_bytes is None:
            log("image_not_found", job_id=job.id, reason="redis_expired_or_missing")
            failed_jobs.append(job)
            continue
        try:
            tensor = loader.preprocess(image_bytes)  # bytes -> (1,1,224,224) tensor
            valid_jobs.append((job, tensor))
        except Exception as e:
            log("preprocess_failed", job_id=job.id, error=str(e))
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
            log("inference_timeout", job_ids=[job.id for job, _ in valid_jobs])
            failed_jobs.extend(job for job, _ in valid_jobs)
            valid_jobs = []
            batch_scores = []
        except Exception as e:
            log("inference_error", job_ids=[job.id for job, _ in valid_jobs], error=str(e))
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
            latency_ms = round((time.time() - batch_start) * 1000, 1)
            log("inference_completed", job_id=job.id, top_label=top_label, latency_ms=latency_ms)

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
    r = redis.from_url(REDIS_URL, decode_responses=True)

    for job in jobs:
        retry_key = f"retry:{job.id}"
        # INCR: 카운터 없으면 0에서 시작해 1 증가, 있으면 +1
        attempt = r.incr(retry_key)
        r.expire(retry_key, 3600)  # 1시간 후 자동 삭제

        if attempt <= settings.MAX_RETRIES:
            # 재시도 가능 -> 큐 맨 뒤에 다시 등록
            log("job_retry", job_id=job.id, attempt=f"{attempt}/{settings.MAX_RETRIES}")
            enqueue(job.id)
        else:
            # 재시도 횟수 소진 -> FAILED 확정
            job.status = InferenceJob.Status.FAILED
            job.save(update_fields=["status", "updated_at"])
            r.delete(retry_key)  # 카운터 정리
            # Dead Letter Queue에 job_id 보관 (운영자가 나중에 확인/재처리 가능)
            r.lpush(DLQ_KEY, job.id)
            # DLQ 크기 상한 1000개 유지 — 무제한 누적으로 인한 메모리 증가 방지
            r.ltrim(DLQ_KEY, 0, 999)
            log("job_failed", job_id=job.id, max_retries=settings.MAX_RETRIES, dlq=DLQ_KEY)


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
        logger.info("⚠️ SIGTERM 수신 — 현재 배치 완료 후 종료")
        shutdown = True

    signal.signal(signal.SIGTERM, handle_sigterm)

    # 모델 로드 (HuggingFace 캐시 또는 다운로드 후 메모리에 올림)
    loader = get_loader()
    loader.load()
    logger.info("✅ 모델 로드 완료 — Worker 준비 ㄱㄱ")

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

        logger.info(f"🔥 Batch 수집: {job_ids}")
        process_batch(job_ids)

    logger.info("✅ Worker 정상 종료")


if __name__ == "__main__":
    run_worker()
