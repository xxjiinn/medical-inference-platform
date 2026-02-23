"""
model_loader.py
역할: AI 모델을 HuggingFace에서 한 번만 로드하고, 전처리 + 추론 기능 제공.
      Spring의 @Bean(singleton)처럼 프로세스 내에서 단 하나의 인스턴스만 존재.
"""

import io
import numpy as np
import torch
import torchxrayvision as xrv
from PIL import Image

# 로드할 모델 이름 (HuggingFace Hub에서 자동 다운로드)
MODEL_WEIGHTS = "densenet121-res224-all"

# 모델 입력 이미지 크기 (torchxrayvision 표준)
IMAGE_SIZE = 224


class ModelLoader:
    """
    AI 모델 싱글톤 클래스.
    Spring의 @Service + @Bean 역할 — 최초 1회만 초기화되고 재사용.
    """

    def __init__(self):
        # 모델 객체 (load() 호출 전까지 None)
        self._model = None

    def load(self):
        """
        모델을 HuggingFace Hub에서 다운로드하고 메모리에 올린다.
        이미 캐시된 경우 HF_HOME 디렉토리에서 바로 로드 (재다운로드 없음).
        """
        print(f"[ModelLoader] Loading model: {MODEL_WEIGHTS}")

        # torchxrayvision이 HuggingFace Hub에서 가중치를 자동 다운로드+캐싱
        self._model = xrv.models.DenseNet(weights=MODEL_WEIGHTS)

        # eval() = 추론 모드 전환 (Spring의 read-only 서비스처럼 Dropout/BatchNorm 비활성화)
        self._model.eval()

        print(f"[ModelLoader] Model loaded. Pathologies: {self._model.pathologies}")

    @property
    def model(self):
        """로드된 모델 반환. 로드 전 접근 시 에러."""
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        return self._model

    def preprocess(self, image_bytes: bytes) -> torch.Tensor:
        """
        클라이언트로부터 받은 이미지 바이트를 모델 입력 텐서로 변환.

        변환 파이프라인:
          bytes -> PIL Image -> numpy(float32) -> normalize -> tensor(1,1,224,224)

        torchxrayvision 입력 규격:
          - 흑백(L채널) 1채널
          - 픽셀값 범위: [-1024, 1024]  (일반 이미지의 [0,1]과 다름!)
          - shape: (batch=1, channel=1, H=224, W=224)
        """
        # 1단계: bytes -> PIL Image (흑백 변환)
        image = Image.open(io.BytesIO(image_bytes)).convert("L")

        # 2단계: PIL -> numpy float32 배열, shape: (H, W)
        img_array = np.array(image, dtype=np.float32)

        # 3단계: torchxrayvision 정규화 — [0,255] -> [-1024, 1024]
        #        xrv.utils.normalize(img, maxval, reshape)
        #        maxval=255 -> 원본 최대값 지정, reshape=True -> (1, H, W)로 자동 변환
        img_normalized = xrv.utils.normalize(img_array, maxval=255, reshape=True)

        # 4단계: 224×224로 리사이즈 (torchxrayvision 내장 변환 사용)
        transform = xrv.datasets.XRayResizer(IMAGE_SIZE)
        img_resized = transform(img_normalized)  # shape: (1, 224, 224)

        # 5단계: numpy -> torch tensor, batch 차원 추가 -> (1, 1, 224, 224)
        tensor = torch.from_numpy(img_resized).unsqueeze(0)

        return tensor

    def predict(self, tensor: torch.Tensor) -> dict:
        """
        전처리된 단일 텐서를 모델에 넣어 18개 질환별 점수를 반환.

        Returns:
            {"Atelectasis": 0.21, "Pneumonia": 0.87, ... (18개)}
        """
        # torch.no_grad(): 역전파(학습) 비활성화 -> 메모리·속도 절약 (추론 전용)
        with torch.no_grad():
            outputs = self._model(tensor)  # shape: (1, 18)

        # 18개 pathology 이름과 점수를 딕셔너리로 묶어 반환
        scores = {
            label: float(score)
            for label, score in zip(self._model.pathologies, outputs[0])
        }
        return scores

    def predict_batch(self, tensors: list[torch.Tensor]) -> list[dict]:
        """
        Micro-batching용: 여러 텐서를 하나의 배치로 묶어 단일 forward pass로 추론.

        개별 추론 N번보다 배치 추론 1번이 효율적인 이유:
          행렬 곱셈은 한번에 처리할수록 CPU/캐시 활용률이 높아짐.
          예: (8, 1, 224, 224) 한번 vs (1, 1, 224, 224) 8번 -> 배치가 ~2-3x 빠름.

        Args:
            tensors: 각 shape (1, 1, 224, 224)인 텐서 리스트

        Returns:
            각 이미지에 대한 scores dict 리스트 (입력 순서와 동일)
        """
        # 개별 텐서 (1,1,224,224) N개를 -> (N,1,224,224) 배치 텐서로 합치기
        # torch.cat: list of tensors를 dim=0 방향으로 이어붙임
        batch_tensor = torch.cat(tensors, dim=0)  # shape: (N, 1, 224, 224)

        with torch.no_grad():
            # 배치 전체를 한 번의 forward pass로 처리 -> shape: (N, 18)
            batch_outputs = self._model(batch_tensor)

        # N개의 결과를 각각 딕셔너리로 변환하여 리스트로 반환
        results = []
        for output in batch_outputs:  # output shape: (18,)
            scores = {
                label: float(score)
                for label, score in zip(self._model.pathologies, output)
            }
            results.append(scores)

        return results


# 프로세스 전역 싱글톤 인스턴스 (Spring의 ApplicationContext에 등록된 Bean과 동일)
_loader = ModelLoader()


def get_loader() -> ModelLoader:
    """어디서든 동일한 ModelLoader 인스턴스를 가져오는 접근자."""
    return _loader
