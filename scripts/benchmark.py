"""
benchmark.py
역할: PyTorch 추론 속도 측정.
      - baseline (torch.compile 미적용) vs compiled (torch.compile 적용) 비교
      - 배치 크기별 latency 측정
      결과는 콘솔에 출력하고 docs/performance.md 작성의 근거 데이터가 됨.
      실행: python scripts/benchmark.py
"""

import sys
import os
import io
import time
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Django 없이 독립 실행 (모델 로더만 사용)
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")

WARMUP_RUNS = 5      # cold-start 제거용 워밍업 횟수
BENCHMARK_RUNS = 50  # 통계적으로 의미있는 p50/p95 측정을 위한 반복 횟수


def make_dummy_bytes() -> bytes:
    """224×224 흑백 더미 이미지 bytes 생성 (벤치마크용)."""
    img = Image.fromarray(np.zeros((224, 224), dtype=np.uint8), mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def percentile_stats(latencies: list[float]) -> dict:
    """latency 리스트에서 p50/p95/p99 (ms 단위) 계산."""
    arr = np.array(latencies) * 1000  # 초 -> ms 변환
    return {
        "p50":  round(float(np.percentile(arr, 50)), 2),
        "p95":  round(float(np.percentile(arr, 95)), 2),
        "p99":  round(float(np.percentile(arr, 99)), 2),
        "mean": round(float(np.mean(arr)), 2),
        "min":  round(float(np.min(arr)), 2),
        "max":  round(float(np.max(arr)), 2),
    }


def benchmark_single(dummy_bytes: bytes, use_compile: bool) -> dict:
    """
    단일 추론 레이턴시 측정.
    use_compile=False: baseline (torch.compile 미적용)
    use_compile=True:  compiled  (torch.compile 적용, 첫 추론 시 컴파일 overhead 발생)

    독립적인 ModelLoader 인스턴스 생성 — 싱글톤 미사용 (두 경우를 공정하게 비교).
    """
    from workers.model_loader import ModelLoader
    loader = ModelLoader(use_compile=use_compile)
    loader.load()

    label = "compiled" if use_compile else "baseline"
    print(f"  Warming up ({label}, {WARMUP_RUNS} runs)...")
    for _ in range(WARMUP_RUNS):
        tensor = loader.preprocess(dummy_bytes)
        loader.predict(tensor)
        # torch.compile의 경우 첫 호출에서 JIT 컴파일 발생
        # 워밍업 중 컴파일이 완료되므로 측정값에 영향 없음

    latencies = []
    print(f"  Benchmarking ({label}, {BENCHMARK_RUNS} runs)...")
    for _ in range(BENCHMARK_RUNS):
        tensor = loader.preprocess(dummy_bytes)
        start = time.perf_counter()
        loader.predict(tensor)
        latencies.append(time.perf_counter() - start)

    return percentile_stats(latencies)


def benchmark_batch(dummy_bytes: bytes, batch_sizes: list[int] = [1, 2, 4, 8]) -> None:
    """배치 크기별 추론 latency 측정 (baseline 기준)."""
    from workers.model_loader import ModelLoader
    loader = ModelLoader(use_compile=False)
    loader.load()

    print("\n=== Batch Inference Latency (baseline, 20 runs each) ===\n")
    print(f"  {'Batch':>5} | {'p50':>8}")
    print("  " + "-" * 18)

    for bs in batch_sizes:
        latencies = []
        for _ in range(20):
            tensors = [loader.preprocess(dummy_bytes) for _ in range(bs)]
            start = time.perf_counter()
            loader.predict_batch(tensors)
            latencies.append(time.perf_counter() - start)
        p50 = np.percentile(latencies, 50) * 1000
        print(f"  {bs:>5} | {p50:>7.1f}ms")


def main():
    dummy_bytes = make_dummy_bytes()

    # 실행 환경 정보 출력
    from workers.model_loader import ModelLoader
    _probe = ModelLoader(use_compile=False)
    device = _probe._device

    print("=" * 60)
    print("  Inference Benchmark: baseline vs torch.compile")
    print("=" * 60)
    print(f"\n  device  : {device}")
    print(f"  PyTorch : {torch.__version__}")
    compile_support = hasattr(torch, "compile")
    print(f"  compile : {'available' if compile_support else 'not available (PyTorch < 2.0)'}")

    # ── baseline 측정 ─────────────────────────────────────────
    print("\n[1] Baseline (no torch.compile)")
    base_stats = benchmark_single(dummy_bytes, use_compile=False)

    # ── compiled 측정 ─────────────────────────────────────────
    if compile_support:
        print("\n[2] torch.compile")
        compiled_stats = benchmark_single(dummy_bytes, use_compile=True)
    else:
        compiled_stats = None
        print("\n[2] torch.compile — SKIPPED (not available)")

    # ── 결과 비교 ─────────────────────────────────────────────
    print("\n=== Results (ms) ===\n")
    if compiled_stats:
        print(f"  {'Metric':>6} | {'Baseline':>10} | {'Compiled':>10} | {'Speedup':>8}")
        print("  " + "-" * 45)
        for key in ["p50", "p95", "p99", "mean", "min", "max"]:
            b = base_stats[key]
            c = compiled_stats[key]
            speedup = b / c if c > 0 else 0
            marker = " ←" if key == "p50" else ""
            print(f"  {key:>6} | {b:>10.2f} | {c:>10.2f} | {speedup:>7.2f}x{marker}")
        print(
            f"\n  Note: torch.compile speedup on CPU is typically 10~30%."
            f"\n  On GPU (CUDA), effect is more pronounced."
        )
    else:
        print(f"  {'Metric':>6} | {'Baseline':>10}")
        print("  " + "-" * 22)
        for key in ["p50", "p95", "p99", "mean", "min", "max"]:
            print(f"  {key:>6} | {base_stats[key]:>10.2f}")

    # ── 배치 스케일링 ─────────────────────────────────────────
    benchmark_batch(dummy_bytes)

    print("\n[Tip] Copy these results to docs/performance.md")


if __name__ == "__main__":
    main()
