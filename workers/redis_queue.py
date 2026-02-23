"""
queue.py
역할: Redis 큐와 캐시 조작을 담당하는 헬퍼 모듈.
      API 서버(enqueue)와 워커(dequeue) 양쪽에서 공유해서 사용.
      Spring의 @Component 유틸리티 빈과 동일한 개념.

Redis 키 구조:
  - 큐:        inference:queue          (List, LPUSH로 넣고 BRPOP으로 꺼냄)
  - 이미지:    image:{sha256}           (Bytes, TTL 600초 — 워커가 꺼내서 추론)
  - 캐시:      cache:sha256:{hash}      (String, TTL 600초 = 10분)
"""

import redis
import os

# 환경변수에서 Redis URL 읽기 (e.g. "redis://redis:6379/0")
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")

# Redis 큐 키 이름
QUEUE_KEY = "inference:queue"

# 중복 요청 캐시 TTL (초)
CACHE_TTL = 600  # 10분


def get_redis() -> redis.Redis:
    """Redis 연결 객체 반환. decode_responses=True로 bytes 대신 str 반환."""
    return redis.from_url(REDIS_URL, decode_responses=True)


def enqueue(job_id: int) -> None:
    """
    job_id를 Redis 큐의 왼쪽에 삽입 (LPUSH).
    워커는 오른쪽에서 꺼냄(BRPOP) -> FIFO 큐 구조.
    Spring의 BlockingQueue.put()과 동일.
    """
    r = get_redis()
    r.lpush(QUEUE_KEY, str(job_id))


def dequeue(timeout: int = 5) -> int | None:
    """
    Redis 큐의 오른쪽에서 job_id를 꺼냄 (BRPOP, blocking).
    큐가 비어있으면 timeout초 동안 대기 후 None 반환.
    Spring의 BlockingQueue.poll(timeout, unit)과 동일.

    Returns:
        job_id (int) or None (타임아웃)
    """
    r = get_redis()
    # BRPOP 반환값: (key, value) 튜플 또는 None
    result = r.brpop(QUEUE_KEY, timeout=timeout)
    if result is None:
        return None
    _, job_id_str = result
    return int(job_id_str)


def get_cache(sha256: str) -> int | None:
    """
    SHA256 해시로 캐시된 job_id를 조회.
    중복 요청 감지용 — 같은 이미지는 재추론하지 않음.

    Returns:
        cached job_id (int) or None (캐시 미스)
    """
    r = get_redis()
    value = r.get(f"cache:sha256:{sha256}")
    return int(value) if value else None


def set_cache(sha256: str, job_id: int) -> None:
    """
    SHA256 -> job_id 매핑을 Redis에 저장 (TTL = CACHE_TTL초).
    추론이 완료된 job을 캐싱해 동일 이미지 재요청 시 빠르게 반환.
    """
    r = get_redis()
    r.set(f"cache:sha256:{sha256}", str(job_id), ex=CACHE_TTL)


def store_image(sha256: str, image_bytes: bytes) -> None:
    """
    이미지 bytes를 Redis에 임시 저장 (TTL = CACHE_TTL초).
    워커가 추론 시 sha256 키로 이미지를 꺼내 사용.
    decode_responses=False로 별도 연결 — bytes 저장을 위해 필요.
    """
    # bytes 저장은 decode_responses=False 연결이 필요
    r = redis.from_url(REDIS_URL, decode_responses=False)
    r.set(f"image:{sha256}", image_bytes, ex=CACHE_TTL)


def collect_batch(max_wait_ms: int = 30, max_size: int = 8) -> list[int]:
    """
    Micro-batching: 첫 job을 BRPOP으로 기다린 뒤,
    max_wait_ms(ms) 동안 추가 job을 non-blocking RPOP으로 더 수집.

    흐름:
      1. BRPOP(blocking, 5s) — 첫 job 올 때까지 대기
      2. 30ms 타임윈도우 내 RPOP 반복 — 추가 job 수집
      3. max_size 초과 시 즉시 반환 (배치 크기 상한)

    Returns:
        job_id 리스트 (비어있으면 큐 타임아웃)
    """
    import time

    r_str = get_redis()                                      # str 응답용 (job_id 읽기)

    # 1단계: 첫 번째 job 블로킹 대기 (최대 5초)
    result = r_str.brpop(QUEUE_KEY, timeout=5)
    if result is None:
        return []  # 5초 동안 job 없음 -> 빈 배치 반환

    _, first_id = result
    job_ids = [int(first_id)]

    # 2단계: 30ms 윈도우 동안 추가 job 수집 (non-blocking RPOP)
    deadline = time.monotonic() + max_wait_ms / 1000.0

    while time.monotonic() < deadline and len(job_ids) < max_size:
        value = r_str.rpop(QUEUE_KEY)  # 큐가 비면 즉시 None 반환 (non-blocking)
        if value is None:
            break  # 현재 큐가 비어있음 — 더 이상 수집할 job 없음
        job_ids.append(int(value))

    return job_ids
