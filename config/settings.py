import os
from pathlib import Path
from dotenv import load_dotenv

# .env 파일을 읽어서 환경변수로 등록 (DB 비밀번호 등 민감정보를 코드 밖에서 관리)
load_dotenv()

# 프로젝트 루트 경로 (config/ 의 부모 디렉토리)
BASE_DIR = Path(__file__).resolve().parent.parent

# Django 암호화에 사용되는 비밀키 (.env에서 필수 로드)
SECRET_KEY = os.environ["SECRET_KEY"]
# True면 디버그 모드 (에러 상세 출력), 운영환경에서는 반드시 False
DEBUG = os.getenv("DEBUG", "False") == "True"
# 허용할 호스트 도메인 (* = 모두 허용, 개발용)
ALLOWED_HOSTS = ["*"]

# Django가 인식할 앱 목록 (Spring의 @ComponentScan 대상)
INSTALLED_APPS = [
    "django.contrib.contenttypes",  # 모델 타입 추적용 Django 내장 앱
    "django.contrib.auth",          # 인증 관련 Django 내장 앱
    "rest_framework",               # Django REST Framework (DRF)
    "apps.jobs",                    # 추론 Job 관리 앱
    "apps.ops",                     # 운영 지표 앱
]

# 요청 처리 전후에 실행되는 미들웨어 (Spring의 Filter/Interceptor)
MIDDLEWARE = [
    "django.middleware.common.CommonMiddleware",  # URL 슬래시 정규화 등 공통 처리
]

# 최상위 URL 설정 파일 위치
ROOT_URLCONF = "config.urls"

# MySQL 데이터베이스 연결 설정 (Spring의 datasource 설정과 동일)
DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.mysql",        # MySQL 드라이버
        "NAME": os.environ["MYSQL_DATABASE"],        # 데이터베이스 이름
        "USER": os.environ["MYSQL_USER"],            # DB 접속 계정
        "PASSWORD": os.environ["MYSQL_PASSWORD"],    # DB 비밀번호
        "HOST": os.environ["MYSQL_HOST"],            # Docker 서비스 이름 "db"
        "PORT": os.environ.get("MYSQL_PORT", "3306"),
        "OPTIONS": {"charset": "utf8mb4"},           # 이모지 포함 유니코드 지원
    }
}

# Redis 연결 URL (큐 + 캐시에 사용)
REDIS_URL = os.environ["REDIS_URL"]

# DRF 기본 설정: 응답을 항상 JSON으로 렌더링
REST_FRAMEWORK = {
    "DEFAULT_RENDERER_CLASSES": ["rest_framework.renderers.JSONRenderer"],
}

# 모델의 기본 PK 타입을 BigInt(64bit)로 설정
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# 추론 관련 설정값 (.env에서 읽음)
WORKER_COUNT = int(os.getenv("WORKER_COUNT", 2))       # 추론 워커 프로세스 수
INFERENCE_TIMEOUT = int(os.getenv("INFERENCE_TIMEOUT", 10))  # 작업당 타임아웃(초)
MAX_RETRIES = int(os.getenv("MAX_RETRIES", 3))         # 실패 시 최대 재시도 횟수
BATCH_WINDOW_MS = int(os.getenv("BATCH_WINDOW_MS", 30))  # 마이크로배치 수집 시간(ms)

# 추론 엔진 선택: "pytorch" 또는 "onnx"
# onnx로 설정 시 convert_to_onnx.py로 변환된 모델 파일이 있어야 함
INFERENCE_ENGINE = os.getenv("INFERENCE_ENGINE", "pytorch")

# ONNX 모델 파일 경로
ONNX_MODEL_PATH = os.getenv(
    "ONNX_MODEL_PATH",
    os.path.join(BASE_DIR, "models", "densenet121.onnx"),
)
