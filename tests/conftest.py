"""
conftest.py
역할: 모든 테스트에서 공유하는 fixtures와 설정 정의.
      Spring의 @TestConfiguration + @BeforeEach 공통 설정과 동일.

주요 역할:
  - DB를 MySQL 대신 SQLite(메모리)로 교체 → 실제 DB 없이 테스트 가능
  - 공통 fixtures 정의 (샘플 이미지, ModelVersion 레코드 등)
"""

import io
import numpy as np
import pytest
from PIL import Image
from django.test import override_settings
from rest_framework.test import APIClient


# ── DB 설정 오버라이드 ───────────────────────────────────────────
# MySQL 대신 SQLite 인메모리 DB 사용 (외부 DB 없이 테스트 가능)
# Spring의 @DataJpaTest가 내부적으로 H2 인메모리 DB로 교체하는 것과 동일
TEST_DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": ":memory:",  # 파일 없이 메모리에만 존재 → 테스트 종료 시 자동 소멸
    }
}


@pytest.fixture(scope="session")
def django_db_setup(django_test_environment, django_db_blocker):
    """
    세션 전체에서 SQLite 인메모리 DB 사용.
    scope="session": 모든 테스트가 DB 인스턴스를 공유 (매 테스트마다 재생성 안 함)
    """
    with override_settings(DATABASES=TEST_DATABASES):
        with django_db_blocker.unblock():
            from django.test.utils import setup_test_environment
            from django.db import connections
            for conn in connections.all():
                conn.ensure_connection()


@pytest.fixture
def api_client():
    """
    DRF APIClient: Django 서버 없이 HTTP 요청을 시뮬레이션.
    Spring의 MockMvc와 동일한 역할.
    """
    return APIClient()


@pytest.fixture
def sample_image_bytes() -> bytes:
    """
    테스트용 더미 흑백 PNG 이미지 bytes 생성.
    실제 X-ray 대신 사용하는 가짜 이미지.
    """
    # 224×224 흑백 이미지 (모든 픽셀 = 0)
    img = Image.fromarray(np.zeros((224, 224), dtype=np.uint8), mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def model_version(db):
    """
    테스트용 ModelVersion DB 레코드 생성.
    jobs 엔드포인트가 모델 조회에 의존하므로 필수 사전 데이터.
    Spring의 @BeforeEach에서 JPA save()로 테스트 데이터 세팅하는 것과 동일.
    """
    from apps.jobs.models import ModelVersion
    return ModelVersion.objects.create(
        name="densenet121-res224-all",
        weights_path="/root/.cache/huggingface/densenet121.pth",
    )
