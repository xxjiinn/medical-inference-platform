"""
warmup.py
역할: 서버 시작 시 더미 이미지로 한 번 추론을 실행해 CPU 캐시를 워밍업.
      첫 번째 실제 요청의 지연 시간(cold-start latency)을 줄이는 것이 목적.
      Spring의 ApplicationRunner / @PostConstruct와 동일한 개념.
"""

import sys
import os
import time
import numpy as np
import torch

# Django 설정 없이 독립 실행 가능하도록 경로 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from workers.model_loader import get_loader


def run_warmup():
    loader = get_loader()

    print("[Warmup] Loading model...")
    # 모델을 메모리에 로드 (HuggingFace 캐시 또는 다운로드)
    loader.load()

    print("[Warmup] Running dummy inference to warm up CPU cache...")
    start = time.time()

    # 224×224 흑백 더미 이미지 생성 (실제 환자 데이터 아님)
    # shape: (1, 1, 224, 224), 값 범위: [-1024, 1024]
    dummy_tensor = torch.zeros(1, 1, 224, 224)

    # 더미 추론 실행 (결과는 버림, 목적은 CPU JIT 컴파일·캐시 워밍업)
    _ = loader.predict(dummy_tensor)

    elapsed = time.time() - start
    print(f"[Warmup] Done. Cold-start inference took {elapsed:.2f}s")
    print("[Warmup] Subsequent requests will be faster.")


if __name__ == "__main__":
    run_warmup()
