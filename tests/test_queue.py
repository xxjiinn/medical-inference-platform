"""
test_queue.py
역할: Redis 큐의 collect_batch 로직을 Mock Redis로 검증.
      실제 Redis 없이 BRPOP/RPOP 동작을 시뮬레이션.
"""

from unittest.mock import patch, MagicMock, call
import pytest

from workers.queue import collect_batch, enqueue, get_cache, set_cache


def make_mock_redis(brpop_result=None, rpop_side_effect=None):
    """
    Mock Redis 객체 생성 헬퍼.
    brpop_result: BRPOP 반환값 (None이면 타임아웃 시뮬레이션)
    rpop_side_effect: RPOP 호출 순서별 반환값 리스트
    """
    r = MagicMock()
    r.brpop.return_value = brpop_result
    if rpop_side_effect:
        r.rpop.side_effect = rpop_side_effect
    else:
        r.rpop.return_value = None  # 기본값: 큐 비어있음
    return r


def test_collect_batch_empty_queue():
    """큐가 비어있을 때 (BRPOP 타임아웃) 빈 리스트를 반환하는지 검증."""
    mock_r = make_mock_redis(brpop_result=None)  # 타임아웃 시뮬레이션

    with patch("workers.queue.get_redis", return_value=mock_r):
        result = collect_batch(max_wait_ms=10, max_size=4)

    assert result == []


def test_collect_batch_single_job():
    """job 1개만 있을 때 [job_id] 반환하는지 검증."""
    # BRPOP: job_id=42 반환, RPOP: 바로 None (더 이상 없음)
    mock_r = make_mock_redis(
        brpop_result=("inference:queue", "42"),
        rpop_side_effect=[None],
    )

    with patch("workers.queue.get_redis", return_value=mock_r):
        result = collect_batch(max_wait_ms=10, max_size=4)

    assert result == [42]


def test_collect_batch_multiple_jobs():
    """30ms 윈도우 내 여러 job이 있을 때 모두 수집하는지 검증."""
    # BRPOP: 첫 번째 job, RPOP: 두 번째, 세 번째 job 순서로 반환
    mock_r = make_mock_redis(
        brpop_result=("inference:queue", "1"),
        rpop_side_effect=["2", "3", None],  # None: 이후 큐 비어있음
    )

    with patch("workers.queue.get_redis", return_value=mock_r):
        result = collect_batch(max_wait_ms=100, max_size=8)

    assert result == [1, 2, 3]


def test_collect_batch_respects_max_size():
    """max_size 초과 시 수집을 멈추는지 검증."""
    # RPOP이 계속 값을 반환해도 max_size=2에서 멈춰야 함
    mock_r = make_mock_redis(
        brpop_result=("inference:queue", "1"),
        rpop_side_effect=["2", "3", "4"],  # 3개 더 있지만 max_size=2라 첫 번째만 가져옴
    )

    with patch("workers.queue.get_redis", return_value=mock_r):
        result = collect_batch(max_wait_ms=100, max_size=2)

    # 첫 번째 (BRPOP) + 두 번째 (RPOP 1회) = 총 2개
    assert len(result) == 2
    assert result == [1, 2]


def test_enqueue_calls_lpush():
    """enqueue()가 Redis LPUSH를 호출하는지 검증."""
    mock_r = MagicMock()

    with patch("workers.queue.get_redis", return_value=mock_r):
        enqueue(99)

    # LPUSH("inference:queue", "99") 호출 여부 확인
    mock_r.lpush.assert_called_once_with("inference:queue", "99")


def test_get_cache_hit():
    """캐시에 값이 있을 때 job_id를 반환하는지 검증."""
    mock_r = MagicMock()
    mock_r.get.return_value = "42"  # 캐시 히트

    with patch("workers.queue.get_redis", return_value=mock_r):
        result = get_cache("abc123")

    assert result == 42  # str → int 변환 확인


def test_get_cache_miss():
    """캐시에 값이 없을 때 None을 반환하는지 검증."""
    mock_r = MagicMock()
    mock_r.get.return_value = None  # 캐시 미스

    with patch("workers.queue.get_redis", return_value=mock_r):
        result = get_cache("abc123")

    assert result is None
