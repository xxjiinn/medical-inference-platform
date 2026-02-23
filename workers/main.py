"""
main.py
역할: 워커 프로세스들을 관리하는 매니저.
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

# 프로젝트 루트 경로 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Django 설정 초기화 (settings.WORKER_COUNT 읽기 위해 필요)
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
import django
django.setup()

from django.conf import settings

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[Manager] %(message)s")


def start_worker_process() -> multiprocessing.Process:
    """
    worker.py의 run_worker()를 새 프로세스로 실행.
    각 프로세스는 독립적으로 모델을 로드하고 Redis 큐를 폴링.
    (프로세스 간 메모리 공유 없음 — PyTorch 멀티프로세싱 충돌 방지)
    """
    from workers.worker import run_worker

    # daemon=False: 메인 프로세스 종료 시 워커가 현재 Job을 완료하고 종료
    p = multiprocessing.Process(target=run_worker, daemon=False)
    p.start()
    logger.info(f"Started worker process PID={p.pid}")
    return p


def run_manager():
    """
    매니저 메인 루프.
    1. WORKER_COUNT개 워커 프로세스 시작
    2. 주기적으로 워커 상태 확인
    3. 크래시된 워커 자동 재시작
    4. SIGTERM 수신 시 모든 워커에 종료 신호 전송 (Graceful Shutdown)
    """
    shutdown = False

    def handle_sigterm(signum, frame):
        """Docker stop 또는 kill 시 SIGTERM 수신 -> 모든 워커 종료 신호 전송."""
        nonlocal shutdown
        logger.info("Received SIGTERM. Shutting down all workers...")
        shutdown = True

    signal.signal(signal.SIGTERM, handle_sigterm)
    signal.signal(signal.SIGINT, handle_sigterm)  # Ctrl+C도 동일하게 처리

    worker_count = settings.WORKER_COUNT
    logger.info(f"Starting {worker_count} worker process(es)...")

    # 초기 워커 프로세스 풀 생성
    # Spring의 ThreadPoolTaskExecutor.setCorePoolSize()와 동일
    processes: list[multiprocessing.Process] = [
        start_worker_process() for _ in range(worker_count)
    ]

    # 매니저 모니터링 루프
    while not shutdown:
        time.sleep(3)  # 3초마다 워커 상태 점검

        for i, p in enumerate(processes):
            if not p.is_alive():
                # 워커가 예기치 않게 종료됨 (크래시) -> 새 프로세스로 교체
                logger.warning(
                    f"Worker {i} (PID={p.pid}) is dead (exit code={p.exitcode}). Restarting..."
                )
                p.close()  # 죽은 프로세스 리소스 해제
                processes[i] = start_worker_process()

    # Graceful Shutdown: 모든 워커에 SIGTERM 전송
    logger.info("Sending SIGTERM to all workers...")
    for p in processes:
        if p.is_alive():
            p.terminate()

    # 각 워커가 현재 Job을 완료하고 종료될 때까지 최대 30초 대기
    for p in processes:
        p.join(timeout=30)
        if p.is_alive():
            # 30초 내 종료 안 되면 강제 종료
            logger.warning(f"Worker PID={p.pid} did not exit in time. Killing.")
            p.kill()

    logger.info("All workers stopped. Manager exiting.")


if __name__ == "__main__":
    # multiprocessing spawn 방식 명시 (Docker Linux 환경 호환성)
    multiprocessing.set_start_method("spawn", force=True)
    run_manager()
