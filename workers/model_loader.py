"""
model_loader.py
역할: AI 모델을 HuggingFace에서 한 번만 로드하고, 전처리 + 추론 기능 제공.
      Spring의 @Bean(singleton)처럼 프로세스 내에서 단 하나의 인스턴스만 존재.
      디바이스 자동 감지 (CUDA → MPS → CPU), torch.compile 선택적 적용.
"""

import io
import os
import logging
import numpy as np
import torch
import torch._dynamo
import torchxrayvision as xrv
from PIL import Image

logger = logging.getLogger(__name__)

# 로드할 모델 이름 (HuggingFace Hub에서 자동 다운로드)
MODEL_WEIGHTS = "densenet121-res224-all"

# 모델 입력 이미지 크기 (torchxrayvision 표준)
IMAGE_SIZE = 224


class ModelLoader:
    """
    AI 모델 싱글톤 클래스.
    Spring의 @Service + @Bean 역할 — 최초 1회만 초기화되고 재사용.
    """

    def __init__(self, use_compile: bool = True):
        self._model = None
        self._pathologies = None   # compile 후에도 안전하게 접근하기 위해 별도 저장
        self._use_compile = use_compile  # torch.compile 적용 여부 (benchmark 비교용)

        # 디바이스 자동 감지: CUDA → MPS(Apple Silicon) → CPU
        # INFERENCE_DEVICE 환경변수로 강제 지정 가능 (예: "cpu", "cuda", "mps")
        requested = os.getenv("INFERENCE_DEVICE", "auto").lower()
        if requested != "auto":
            self._device = torch.device(requested)
        elif torch.cuda.is_available():
            self._device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            # Apple Silicon GPU (Metal Performance Shaders)
            self._device = torch.device("mps")
        else:
            self._device = torch.device("cpu")

    def load(self):
        """
        모델을 HuggingFace Hub에서 다운로드하고 지정 디바이스 메모리에 올린다.
        이미 캐시된 경우 HF_HOME 디렉토리에서 바로 로드 (재다운로드 없음).
        """
        logger.info(f"☑️ 모델 로딩 중: {MODEL_WEIGHTS} → device={self._device}")

        # torchxrayvision이 HuggingFace Hub에서 가중치를 자동 다운로드 + 메모리에 캐싱
        self._model = xrv.models.DenseNet(weights=MODEL_WEIGHTS)

        # 모델을 지정 디바이스로 이동 (CPU/CUDA/MPS)
        # GPU 환경에서는 이 시점에 VRAM에 ~110MB 적재
        self._model.to(self._device)

        # eval() = 추론 모드 전환 (Spring의 read-only 서비스처럼 Dropout/BatchNorm 비활성화)
        self._model.eval()

        # pathologies 속성을 compile 전에 미리 저장.
        # torch.compile은 모델을 OptimizedModule로 래핑하는데,
        # 래퍼를 통한 비텐서 속성 접근이 불안정할 수 있어 리스트로 고정.
        self._pathologies = list(self._model.pathologies)

        # torch.compile 시도 (PyTorch 2.0+)
        # ONNX는 정적 그래프 제약(NonZero+GatherND)으로 변환 불가였으나,
        # torch.compile은 동적 형태(data-dependent shape)를 지원하므로 적용 가능.
        # 첫 번째 추론 시 JIT 컴파일 발생 → 이후 호출부터 최적화된 커널 실행.
        if self._use_compile:
            if not hasattr(torch, "compile"):
                logger.warning("⚠️ torch.compile 미지원 — PyTorch 2.0 이상 필요")
            else:
                try:
                    # C++ 컴파일러 없는 환경(최소 Docker 이미지 등)에서
                    # 첫 forward pass 시 컴파일 실패 시 자동으로 eager mode로 fallback
                    torch._dynamo.config.suppress_errors = True
                    self._model = torch.compile(self._model)
                    logger.info("✅ torch.compile 적용 — 첫 추론 시 컴파일, 이후 최적화 효과")
                    logger.info("   (g++ 없는 환경에서는 자동으로 eager mode fallback)")
                except Exception as e:
                    logger.warning(f"⚠️ torch.compile 미적용 ({type(e).__name__}): {e}")

        logger.info(
            f"✅ 모델 로드 완료 — 병리 항목 수: {len(self._pathologies)}, "
            f"device={self._device}, compiled={self._use_compile}"
        )

    @property
    def model(self):
        """로드된 모델 반환. 로드 전 접근 시 에러."""
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        return self._model

    def preprocess(self, image_bytes: bytes) -> torch.Tensor:
        """
        클라이언트로부터 받은 이미지 바이트를 CPU 텐서로 변환.
        디바이스 이동은 predict() 시점에 수행 (GPU VRAM은 추론 직전에만 점유).

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

        # 5단계: numpy -> CPU 텐서, batch 차원 추가 -> (1, 1, 224, 224)
        # CPU에 두는 이유: 배치 수집 완료 후 predict_batch()에서 디바이스로 한 번에 전송
        tensor = torch.from_numpy(img_resized).unsqueeze(0)
        return tensor

    def predict(self, tensor: torch.Tensor) -> dict:
        """
        전처리된 단일 텐서를 모델에 넣어 18개 질환별 점수를 반환.

        Returns:
            {"Atelectasis": 0.21, "Pneumonia": 0.87, ... (18개)}
        """
        with torch.no_grad():
            # 텐서를 모델 디바이스로 이동 후 추론 (CPU→GPU 전송 포함)
            outputs = self._model(tensor.to(self._device))  # shape: (1, 18)

        scores = {
            label: float(score)
            for label, score in zip(self._pathologies, outputs[0])
        }
        return scores

    def predict_batch(self, tensors: list[torch.Tensor]) -> list[dict]:
        """
        Micro-batching용: 여러 텐서를 하나의 배치로 묶어 단일 forward pass로 추론.

        GPU 환경에서는 배치 크기가 늘어도 선형 이상의 처리량 향상이 가능.
        GPU 행렬 연산은 배치 단위 병렬처리를 지원 — CPU와 달리 선형 스케일링 아님.
        (예: bs=8에서 GPU는 ~2x, CPU는 ~8x 시간 소요)

        Args:
            tensors: 각 shape (1, 1, 224, 224)인 CPU 텐서 리스트

        Returns:
            각 이미지에 대한 scores dict 리스트 (입력 순서와 동일)
        """
        # 개별 CPU 텐서를 (N,1,224,224)로 합친 뒤 디바이스로 한 번에 전송
        # (개별 전송보다 배치 전송이 Host→Device 메모리 복사 횟수를 줄임)
        batch_tensor = torch.cat(tensors, dim=0).to(self._device)

        with torch.no_grad():
            # 배치 전체를 한 번의 forward pass로 처리 -> shape: (N, 18)
            batch_outputs = self._model(batch_tensor)

        results = []
        for output in batch_outputs:  # output shape: (18,)
            scores = {
                label: float(score)
                for label, score in zip(self._pathologies, output)
            }
            results.append(scores)

        return results


# 프로세스 전역 싱글톤 인스턴스 (Spring의 ApplicationContext에 등록된 Bean과 동일)
_loader = ModelLoader(use_compile=True)


def get_loader() -> ModelLoader:
    """어디서든 동일한 ModelLoader 인스턴스를 가져오는 접근자."""
    return _loader
