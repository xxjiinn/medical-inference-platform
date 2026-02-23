"""
benchmark.py
역할: PyTorch vs ONNX Runtime의 추론 속도를 측정하고 p50/p95/p99 레이턴시를 비교.
      결과는 콘솔에 출력하고 docs/performance.md 작성의 근거 데이터가 됨.
      실행: python scripts/benchmark.py
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Django 없이 독립 실행 (모델 로더만 사용)
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")

WARMUP_RUNS = 5    # 측정 전 워밍업 실행 횟수 (cold-start 제거 목적)
BENCHMARK_RUNS = 50  # 실제 측정 반복 횟수


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


def benchmark_pytorch(dummy_bytes: bytes) -> dict:
    """PyTorch 엔진 단일 추론 레이턴시 측정."""
    from workers.model_loader import get_loader
    loader = get_loader()
    loader.load()

    # 워밍업: CPU 캐시 + JIT 컴파일 준비
    print(f"  Warming up PyTorch ({WARMUP_RUNS} runs)...")
    for _ in range(WARMUP_RUNS):
        tensor = loader.preprocess(dummy_bytes)
        loader.predict(tensor)

    # 실제 측정
    latencies = []
    print(f"  Benchmarking PyTorch ({BENCHMARK_RUNS} runs)...")
    for _ in range(BENCHMARK_RUNS):
        tensor = loader.preprocess(dummy_bytes)
        start = time.perf_counter()   # 고정밀 타이머 시작
        loader.predict(tensor)
        elapsed = time.perf_counter() - start  # 추론 시간만 측정 (전처리 제외)
        latencies.append(elapsed)

    return percentile_stats(latencies)


def benchmark_onnx(dummy_bytes: bytes) -> dict:
    """ONNX Runtime 엔진 단일 추론 레이턴시 측정."""
    from workers.onnx_loader import get_onnx_loader
    loader = get_onnx_loader()
    loader.load()

    # 워밍업
    print(f"  Warming up ONNX ({WARMUP_RUNS} runs)...")
    for _ in range(WARMUP_RUNS):
        arr = loader.preprocess(dummy_bytes)
        loader.predict(arr)

    # 실제 측정
    latencies = []
    print(f"  Benchmarking ONNX ({BENCHMARK_RUNS} runs)...")
    for _ in range(BENCHMARK_RUNS):
        arr = loader.preprocess(dummy_bytes)
        start = time.perf_counter()
        loader.predict(arr)
        elapsed = time.perf_counter() - start
        latencies.append(elapsed)

    return percentile_stats(latencies)


def benchmark_batch(batch_sizes: list[int] = [1, 2, 4, 8]) -> None:
    """PyTorch vs ONNX 배치 추론 레이턴시 비교 (배치 크기별)."""
    from workers.model_loader import get_loader
    from workers.onnx_loader import get_onnx_loader
    import io
    from PIL import Image

    # 더미 이미지 bytes 생성 (224×224 흑백)
    dummy_img = Image.fromarray(np.zeros((224, 224), dtype=np.uint8), mode="L")
    buf = io.BytesIO()
    dummy_img.save(buf, format="PNG")
    dummy_bytes = buf.getvalue()

    pt_loader = get_loader()
    pt_loader.load()
    onnx_loader = get_onnx_loader()
    onnx_loader.load()

    print("\n=== Batch Inference Benchmark ===")
    print(f"{'Batch':>6} | {'PyTorch p50':>12} | {'ONNX p50':>10} | {'Speedup':>8}")
    print("-" * 50)

    for bs in batch_sizes:
        # PyTorch 배치 측정
        pt_latencies = []
        for _ in range(20):
            tensors = [pt_loader.preprocess(dummy_bytes) for _ in range(bs)]
            start = time.perf_counter()
            pt_loader.predict_batch(tensors)
            pt_latencies.append(time.perf_counter() - start)

        # ONNX 배치 측정 시도
        try:
            onnx_latencies = []
            for _ in range(20):
                arrays = [onnx_loader.preprocess(dummy_bytes) for _ in range(bs)]
                start = time.perf_counter()
                onnx_loader.predict_batch(arrays)
                onnx_latencies.append(time.perf_counter() - start)
            onnx_p50 = np.percentile(onnx_latencies, 50) * 1000
        except Exception:
            onnx_p50 = None

        pt_p50 = np.percentile(pt_latencies, 50) * 1000
        if onnx_p50:
            speedup = pt_p50 / onnx_p50
            print(f"{bs:>6} | {pt_p50:>10.1f}ms | {onnx_p50:>8.1f}ms | {speedup:>7.2f}x")
        else:
            print(f"{bs:>6} | {pt_p50:>10.1f}ms | {'N/A':>8} | {'N/A':>7}")


def main():
    import io
    from PIL import Image

    # 더미 이미지 bytes 생성
    dummy_img = Image.fromarray(np.zeros((224, 224), dtype=np.uint8), mode="L")
    buf = io.BytesIO()
    dummy_img.save(buf, format="PNG")
    dummy_bytes = buf.getvalue()

    print("=" * 55)
    print("  Single Inference Benchmark: PyTorch vs ONNX Runtime")
    print("=" * 55)

    # PyTorch 측정
    print("\n[PyTorch]")
    pt_stats = benchmark_pytorch(dummy_bytes)

    # ONNX 측정 시도
    print("\n[ONNX Runtime]")
    try:
        onnx_stats = benchmark_onnx(dummy_bytes)
        onnx_available = True
    except Exception as e:
        # densenet121-res224-all 모델은 data-dependent control flow(NonZero+GatherND)를 사용해
        # ONNX 정적 그래프로 표현 불가. 이는 모델 아키텍처의 근본적인 제약임.
        print(f"  [SKIP] ONNX inference failed: {type(e).__name__}")
        print("  Reason: densenet121-res224-all uses NonZero+GatherND for multi-dataset")
        print("  pathology aggregation — output shapes are data-dependent and cannot be")
        print("  represented in ONNX's static computation graph.")
        onnx_available = False

    # 결과 출력
    print("\n=== Results (ms) ===")
    if onnx_available:
        print(f"{'Metric':>8} | {'PyTorch':>10} | {'ONNX':>10} | {'Speedup':>8}")
        print("-" * 45)
        for key in ["p50", "p95", "p99", "mean", "min", "max"]:
            pt_val = pt_stats[key]
            onnx_val = onnx_stats[key]
            speedup = pt_val / onnx_val if onnx_val > 0 else 0
            print(f"{key:>8} | {pt_val:>10.2f} | {onnx_val:>10.2f} | {speedup:>7.2f}x")
    else:
        print(f"{'Metric':>8} | {'PyTorch':>10}")
        print("-" * 25)
        for key in ["p50", "p95", "p99", "mean", "min", "max"]:
            print(f"{key:>8} | {pt_stats[key]:>10.2f}")
        print("\n  Note: ONNX not available for this model (see above).")

    # 배치 비교 (PyTorch only)
    benchmark_batch()

    print("\n[Tip] Copy these results to docs/performance.md")


if __name__ == "__main__":
    main()
