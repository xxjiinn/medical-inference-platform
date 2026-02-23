"""
convert_to_onnx.py
역할: 학습된 PyTorch 모델을 ONNX 포맷으로 변환하여 저장.
      변환은 1회만 실행하면 되며, 이후 ONNX Runtime으로 더 빠른 추론 가능.
      실행: python scripts/convert_to_onnx.py
"""

import sys
import os
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from workers.model_loader import get_loader

# ONNX 파일 저장 경로
OUTPUT_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "densenet121.onnx")


def convert():
    # 저장 디렉토리 생성 (없으면)
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    print("[Convert] Loading PyTorch model...")
    loader = get_loader()
    loader.load()  # HuggingFace에서 모델 로드

    # ONNX 변환용 더미 입력 (실제 추론과 동일한 shape: 1×1×224×224)
    # torch.onnx.export는 실제 데이터로 모델을 1회 실행하며 계산 그래프를 추출
    dummy_input = torch.zeros(1, 1, 224, 224)

    print(f"[Convert] Exporting to ONNX: {OUTPUT_PATH}")
    torch.onnx.export(
        loader.model,           # 변환할 PyTorch 모델
        dummy_input,            # 입력 shape 정의용 더미 텐서
        OUTPUT_PATH,            # 저장 경로
        export_params=True,     # 가중치(파라미터)도 함께 저장
        opset_version=11,       # opset 11 = 가장 안정적인 버전
        do_constant_folding=False, # True로 설정 시 DenseNet Reshape 노드 오류 발생 -> 비활성화
        input_names=["input"],  # 입력 노드 이름 (ONNX Runtime에서 참조)
        output_names=["output"], # 출력 노드 이름
        dynamic_axes={
            # batch 차원을 동적으로 설정 -> 배치 크기가 달라져도 동작
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )

    print(f"[Convert] Done. File size: {os.path.getsize(OUTPUT_PATH) / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    convert()
