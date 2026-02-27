"""
main.py
역할: Worker 프로세스들을 관리하는 매니저.
      WORKER_COUNT만큼 worker.py를 별도 프로세스로 실행하고,
      크래시 발생 시 자동으로 재시작 (Supervisor 역할).
      Spring의 ThreadPoolTaskExecutor 관리자와 유사한 개념.
"""

import os
import sys
import time
import signal
import logging
import multiprocessing
from datetime import timedelta

import redis

# 프로젝트 루트 경로 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Django 설정 초기화 (settings.WORKER_COUNT 읽기 위해 필요)
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
import django
django.setup()

from django.conf import settings

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[Manager] %(message)s")


def _recover_stuck_jobs() -> None:
    """
    두 가지 stuck 상태를 복구:

    1. IN_PROGRESS stuck (10분 초과):
       워커 크래시(SIGKILL 등)로 추론 중 멈춘 job.
       기준: updated_at이 10분 이상 지난 IN_PROGRESS job.

    2. QUEUED stuck (5분 초과):
       API 서버 크래시로 DB 생성 후 Redis enqueue가 유실된 job.
       (정상 흐름에서 QUEUED → IN_PROGRESS는 수초 이내 — 5분 경과 시 유실 판단)
       image Redis TTL = 10분이므로 5분 기준 복구 시 이미지 여전히 유효.

    두 케이스 모두 동일한 retry 카운터(retry:{job_id})를 공유.
    MAX_RETRIES 초과 시 FAILED + DLQ 처리.
    """
    from django.utils import timezone
    from apps.jobs.models import InferenceJob
    from workers.redis_queue import enqueue, REDIS_URL, DLQ_KEY

    now = timezone.now()
    # IN_PROGRESS: updated_at 기준 (마지막 상태 변경 시각)
    in_progress_threshold = now - timedelta(minutes=10)
    # QUEUED stuck: created_at 기준 (생성 이후 한 번도 처리 시작 안 됨)
    queued_threshold = now - timedelta(minutes=5)

    stuck_in_progress = list(InferenceJob.objects.filter(
        status=InferenceJob.Status.IN_PROGRESS,
        updated_at__lt=in_progress_threshold,
    ))
    stuck_queued = list(InferenceJob.objects.filter(
        status=InferenceJob.Status.QUEUED,
        created_at__lt=queued_threshold,
    ))

    if not stuck_in_progress and not stuck_queued:
        return

    r = redis.from_url(REDIS_URL, decode_responses=True)

    # ── IN_PROGRESS stuck 복구 ────────────────────────────────────
    if stuck_in_progress:
        logger.warning(f"❗️ IN_PROGRESS stuck job {len(stuck_in_progress)}개 감지")
        for job in stuck_in_progress:
            retry_key = f"retry:{job.id}"
            attempt = r.incr(retry_key)   # 복구 시도 횟수 증가
            r.expire(retry_key, 3600)     # 1시간 후 자동 삭제

            if attempt > settings.MAX_RETRIES:
                job.status = InferenceJob.Status.FAILED
                job.save(update_fields=["status", "updated_at"])
                r.delete(retry_key)
                r.lpush(DLQ_KEY, job.id)
                r.ltrim(DLQ_KEY, 0, 999)  # DLQ 상한 1000개 유지
                logger.warning(
                    f"  ❌ Job {job.id} 재시도 {settings.MAX_RETRIES}회 초과 → FAILED (DLQ)"
                )
            else:
                job.status = InferenceJob.Status.QUEUED
                job.save(update_fields=["status", "updated_at"])
                enqueue(job.id)
                logger.info(f"  ↩️  Job {job.id} 재큐잉 ({attempt}/{settings.MAX_RETRIES})")

    # ── QUEUED stuck 복구 ──────────────────────────────────────────
    # 원인: POST /v1/jobs에서 DB create 성공 후 enqueue 전 서버 크래시
    # 해결: Redis 큐에 job_id 재등록 (DB status는 이미 QUEUED이므로 변경 불필요)
    if stuck_queued:
        logger.warning(f"❗️ QUEUED stuck job {len(stuck_queued)}개 감지 (enqueue 유실 추정)")
        for job in stuck_queued:
            retry_key = f"retry:{job.id}"
            attempt = r.incr(retry_key)   # 복구 시도 횟수 증가
            r.expire(retry_key, 3600)     # 1시간 후 자동 삭제

            if attempt > settings.MAX_RETRIES:
                job.status = InferenceJob.Status.FAILED
                job.save(update_fields=["status", "updated_at"])
                r.delete(retry_key)
                r.lpush(DLQ_KEY, job.id)
                logger.warning(
                    f"  ❌ QUEUED stuck Job {job.id} → FAILED (DLQ)"
                )
            else:
                # DB status는 QUEUED 유지 — Redis 큐에만 재등록
                enqueue(job.id)
                logger.info(
                    f"  ↩️  QUEUED stuck Job {job.id} 재큐잉 ({attempt}/{settings.MAX_RETRIES})"
                )


def start_worker_process() -> multiprocessing.Process:
    """
    worker.py의 run_worker()를 새 프로세스로 실행.
    각 프로세스는 독립적으로 모델을 로드하고 Redis 큐를 폴링.
    (프로세스 간 메모리 공유 없음 — PyTorch 멀티프로세싱 충돌 방지)
    """
    from workers.worker import run_worker

    # daemon=False: 메인 프로세스 종료 시 Worker가 현재 Job을 완료하고 종료
    p = multiprocessing.Process(target=run_worker, daemon=False)
    p.start()
    logger.info(f"✅ Worker 프로세스 시작 — PID={p.pid}")
    return p


def run_manager():
    """
    매니저 메인 루프.
    1. WORKER_COUNT개 Worker 프로세스 시작
    2. 주기적으로 Worker 상태 확인
    3. 크래시된 Worker 자동 재시작
    4. SIGTERM 수신 시 모든 Worker에 종료 신호 전송 (Graceful Shutdown)
    """
    shutdown = False

    def handle_sigterm(signum, frame):
        """Docker stop 또는 kill 시 SIGTERM 수신 -> 모든 Worker 종료 신호 전송."""
        nonlocal shutdown
        logger.info("⚠️ SIGTERM 수신 — 모든 Worker 종료 시작")
        shutdown = True

    signal.signal(signal.SIGTERM, handle_sigterm)
    signal.signal(signal.SIGINT, handle_sigterm)  # Ctrl+C도 동일하게 처리

    worker_count = settings.WORKER_COUNT
    logger.info(f"🔥 매니저 시작 — Worker {worker_count}개 실행")

    # 초기 Worker 프로세스 풀 생성
    # Spring의 ThreadPoolTaskExecutor.setCorePoolSize()와 동일
    processes: list[multiprocessing.Process] = [
        start_worker_process() for _ in range(worker_count)
    ]

    # stuck job 복구 타이머 (10분마다 실행)
    RECOVERY_INTERVAL = 600
    _last_recovery = time.monotonic()

    # 매니저 모니터링 루프
    while not shutdown:
        time.sleep(3)  # 3초마다 Worker 상태 점검

        for i, p in enumerate(processes):
            if not p.is_alive():
                # Worker가 예기치 않게 종료됨 (크래시) -> 새 프로세스로 교체
                logger.warning(
                    f"❗️ Worker {i} 크래시 감지 (PID={p.pid}, exit={p.exitcode}) — 재시작"
                )
                p.close()  # 죽은 프로세스 리소스 해제
                processes[i] = start_worker_process()

        # 10분마다 stuck job 복구 실행 (IN_PROGRESS 10분↑ + QUEUED 5분↑)
        if time.monotonic() - _last_recovery >= RECOVERY_INTERVAL:
            _recover_stuck_jobs()
            _last_recovery = time.monotonic()

    # Graceful Shutdown: 모든 Worker에 SIGTERM 전송
    logger.info("⚠️ 모든 Worker에 SIGTERM 전송 중...")
    for p in processes:
        if p.is_alive():
            p.terminate()

    # 각 Worker가 현재 Job을 완료하고 종료될 때까지 최대 30초 대기
    for p in processes:
        p.join(timeout=30)
        if p.is_alive():
            # 30초 내 종료 안 되면 강제 종료
            logger.warning(f"❌ Worker PID={p.pid} 30초 내 미종료 — 강제 종료")
            p.kill()

    logger.info("✅ 모든 Worker 종료 완료 — 매니저 종료")


if __name__ == "__main__":
    # multiprocessing spawn 방식 명시 (Docker Linux 환경 호환성)
    multiprocessing.set_start_method("spawn", force=True)
    run_manager()
