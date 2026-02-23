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
from rest_framework.test import APIClient


@pytest.fixture(scope="session")
def django_db_setup(django_test_environment, django_db_blocker):
    """
    세션 전체에서 SQLite 인메모리 DB 사용.
    pytest.ini의 DJANGO_SETTINGS_MODULE = config.test_settings 으로
    DB가 이미 SQLite로 설정돼 있으므로 여기서는 migrate만 실행.
    scope="session": 모든 테스트가 DB 인스턴스를 공유 (매 테스트마다 재생성 안 함)
    """
    with django_db_blocker.unblock():
        # SQLite 인메모리 DB에 테이블 생성 (Spring의 Flyway/Liquibase와 동일)
        from django.core.management import call_command
        call_command("migrate", verbosity=0)


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
