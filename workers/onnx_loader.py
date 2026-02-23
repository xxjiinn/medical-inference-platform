"""
onnx_loader.py
역할: ONNX Runtime을 사용한 추론 로더.
      model_loader.py(PyTorch)와 완전히 동일한 인터페이스 제공 ->
      worker.py에서 엔진만 교체하면 나머지 코드 변경 없이 동작.
      Spring의 인터페이스 기반 DI 전환과 동일한 개념.
"""

import io
import os
import numpy as np
import onnxruntime as ort
import torchxrayvision as xrv
from PIL import Image

# ONNX 모델 파일 경로 (convert_to_onnx.py로 생성한 파일)
DEFAULT_ONNX_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models", "densenet121.onnx"
)

IMAGE_SIZE = 224

# torchxrayvision DenseNet의 18개 질환 레이블 (PyTorch 모델과 동일한 순서)
PATHOLOGIES = [
    "Atelectasis", "Consolidation", "Infiltration", "Pneumothorax",
    "Edema", "Emphysema", "Fibrosis", "Effusion", "Pneumonia",
    "Pleural_Thickening", "Cardiomegaly", "Nodule", "Mass", "Hernia",
    "Lung Lesion", "Fracture", "Lung Opacity", "Enlarged Cardiomediastinum",
]


class OnnxLoader:
    """
    ONNX Runtime 기반 추론 싱글톤.
    model_loader.py의 ModelLoader와 동일한 메서드 시그니처를 가짐.
    """

    def __init__(self, onnx_path: str = DEFAULT_ONNX_PATH):
        self._session = None          # ONNX Runtime 세션 (load() 전 None)
        self._input_name = None       # 모델 입력 노드 이름
        self._onnx_path = onnx_path

    def load(self):
        """
        ONNX 모델 파일을 읽어 InferenceSession 생성.
        SessionOptions로 CPU 스레드 수 최적화.
        """
        if not os.path.exists(self._onnx_path):
            raise FileNotFoundError(
                f"ONNX model not found: {self._onnx_path}\n"
                "Run: python scripts/convert_to_onnx.py"
            )

        # ONNX Runtime CPU 실행 옵션 설정
        opts = ort.SessionOptions()
        # intra_op: 단일 연산 내부 병렬 스레드 수 (행렬곱 등)
        opts.intra_op_num_threads = int(os.getenv("ORT_INTRA_THREADS", 4))
        # inter_op: 연산 간 병렬 스레드 수 (순차 실행이 CPU에서 더 효율적)
        opts.inter_op_num_threads = 1
        # 순차 실행 모드: CPU에서 연산 간 오버헤드 최소화
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        # CPUExecutionProvider: CPU 전용 실행 (GPU 없는 환경)
        self._session = ort.InferenceSession(
            self._onnx_path,
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )

        # 입력 노드 이름 저장 (convert_to_onnx.py에서 "input"으로 지정)
        self._input_name = self._session.get_inputs()[0].name
        print(f"[OnnxLoader] Session ready. Input: {self._input_name}")

    def preprocess(self, image_bytes: bytes) -> np.ndarray:
        """
        이미지 bytes -> ONNX Runtime 입력 numpy 배열 변환.
        PyTorch와 동일한 전처리 파이프라인, 반환 타입만 ndarray로 다름.
        """
        # bytes -> PIL (흑백)
        image = Image.open(io.BytesIO(image_bytes)).convert("L")
        # PIL -> numpy float32
        img_array = np.array(image, dtype=np.float32)
        # [0,255] -> [-1024, 1024] 정규화 + (1, H, W) reshape
        img_normalized = xrv.utils.normalize(img_array, maxval=255, reshape=True)
        # 224×224 리사이즈
        transform = xrv.datasets.XRayResizer(IMAGE_SIZE)
        img_resized = transform(img_normalized)  # shape: (1, 224, 224)
        # batch 차원 추가 -> (1, 1, 224, 224), ONNX Runtime은 numpy 필요
        return img_resized[np.newaxis, :]  # shape: (1, 1, 224, 224)

    def predict(self, inputs: np.ndarray) -> dict:
        """
        단일 numpy 배열 추론 -> 18개 질환 점수 딕셔너리 반환.
        """
        # ONNX Runtime 실행: {입력노드명: numpy배열} 딕셔너리로 전달
        outputs = self._session.run(None, {self._input_name: inputs})
        # outputs[0] shape: (1, 18) -> [0]으로 첫 번째 배치 결과 추출
        scores = {
            label: float(score)
            for label, score in zip(PATHOLOGIES, outputs[0][0])
        }
        return scores

    def predict_batch(self, inputs_list: list[np.ndarray]) -> list[dict]:
        """
        Micro-batching용: 여러 numpy 배열을 배치로 묶어 단일 ONNX 추론.
        """
        # 개별 (1,1,224,224) 배열 N개 -> (N,1,224,224) 배치 배열
        batch = np.concatenate(inputs_list, axis=0)

        # 배치 추론 실행
        outputs = self._session.run(None, {self._input_name: batch})
        # outputs[0] shape: (N, 18)

        results = []
        for row in outputs[0]:  # row shape: (18,)
            scores = {
                label: float(score)
                for label, score in zip(PATHOLOGIES, row)
            }
            results.append(scores)

        return results


# 프로세스 전역 싱글톤
_onnx_loader = OnnxLoader()


def get_onnx_loader() -> OnnxLoader:
    """어디서든 동일한 OnnxLoader 인스턴스를 반환."""
    return _onnx_loader
