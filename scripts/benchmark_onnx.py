"""
benchmark_onnx.py
역할: PyTorch baseline vs ONNX Runtime 추론 속도 비교 측정.
      ONNX 모델이 없으면 자동으로 변환 후 벤치마크 실행.
      실행: python scripts/benchmark_onnx.py
"""

import sys
import os
import io
import time
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")

WARMUP_RUNS = 5       # cold-start 제거용 워밍업 횟수
BENCHMARK_RUNS = 50   # p50/p95/p99 통계를 위한 반복 횟수

# ONNX 모델 저장 경로 (convert_to_onnx.py와 동일)
ONNX_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models", "densenet121.onnx"
)


def make_dummy_bytes() -> bytes:
    """224×224 흑백 더미 이미지 bytes 생성."""
    img = Image.fromarray(np.zeros((224, 224), dtype=np.uint8), mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def percentile_stats(latencies: list[float]) -> dict:
    """latency 리스트에서 p50/p95/p99 (ms 단위) 계산."""
    arr = np.array(latencies) * 1000  # 초 → ms
    return {
        "p50":  round(float(np.percentile(arr, 50)), 2),
        "p95":  round(float(np.percentile(arr, 95)), 2),
        "p99":  round(float(np.percentile(arr, 99)), 2),
        "mean": round(float(np.mean(arr)), 2),
    }


def benchmark_pytorch(dummy_bytes: bytes) -> dict:
    """PyTorch baseline 단일 추론 latency 측정 (torch.compile 미적용)."""
    from workers.model_loader import ModelLoader
    loader = ModelLoader(use_compile=False)  # 공정한 비교를 위해 compile 미적용
    loader.load()

    print(f"  Warming up (pytorch, {WARMUP_RUNS} runs)...")
    for _ in range(WARMUP_RUNS):
        loader.predict(loader.preprocess(dummy_bytes))

    latencies = []
    print(f"  Benchmarking (pytorch, {BENCHMARK_RUNS} runs)...")
    for _ in range(BENCHMARK_RUNS):
        tensor = loader.preprocess(dummy_bytes)
        start = time.perf_counter()
        loader.predict(tensor)
        latencies.append(time.perf_counter() - start)

    return percentile_stats(latencies)


def benchmark_onnx(dummy_bytes: bytes) -> dict:
    """ONNX Runtime 단일 추론 latency 측정."""
    from workers.onnx_loader import OnnxLoader
    loader = OnnxLoader(onnx_path=ONNX_PATH)
    loader.load()

    print(f"  Warming up (onnx, {WARMUP_RUNS} runs)...")
    for _ in range(WARMUP_RUNS):
        loader.predict(loader.preprocess(dummy_bytes))

    latencies = []
    print(f"  Benchmarking (onnx, {BENCHMARK_RUNS} runs)...")
    for _ in range(BENCHMARK_RUNS):
        inputs = loader.preprocess(dummy_bytes)
        start = time.perf_counter()
        loader.predict(inputs)
        latencies.append(time.perf_counter() - start)

    return percentile_stats(latencies)


def ensure_onnx_model():
    """ONNX 모델 파일이 없으면 자동으로 변환."""
    if os.path.exists(ONNX_PATH):
        print(f"  ONNX 모델 확인: {ONNX_PATH}")
        return
    print("  ONNX 모델 없음 — 자동 변환 시작...")
    # convert_to_onnx.py의 convert() 함수 직접 호출
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from convert_to_onnx import convert
    convert()


def main():
    print("=" * 60)
    print("  Inference Benchmark: PyTorch vs ONNX Runtime")
    print("=" * 60)
    print(f"\n  PyTorch : {torch.__version__}")
    print(f"  ONNX    : {ONNX_PATH}")

    # ONNX 모델 준비
    ensure_onnx_model()

    dummy_bytes = make_dummy_bytes()

    # ── PyTorch baseline 측정 ──────────────────────────────────
    print("\n[1] PyTorch (no torch.compile)")
    pytorch_stats = benchmark_pytorch(dummy_bytes)

    # ── ONNX Runtime 측정 ─────────────────────────────────────
    print("\n[2] ONNX Runtime")
    onnx_stats = benchmark_onnx(dummy_bytes)

    # ── 결과 비교 ─────────────────────────────────────────────
    print("\n=== Results (ms) ===\n")
    print(f"  {'Metric':>6} | {'PyTorch':>10} | {'ONNX':>10} | {'Speedup':>8}")
    print("  " + "-" * 45)
    for key in ["p50", "p95", "p99", "mean"]:
        p = pytorch_stats[key]
        o = onnx_stats[key]
        speedup = p / o if o > 0 else 0  # >1.0이면 ONNX가 더 빠름
        marker = " ←" if key == "p50" else ""
        print(f"  {key:>6} | {p:>10.2f} | {o:>10.2f} | {speedup:>7.2f}x{marker}")

    print("\n[Tip] Copy these results to docs/performance.md")


if __name__ == "__main__":
    main()
