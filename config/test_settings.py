"""
test_settings.py
역할: 테스트 전용 Django 설정.
      config.settings를 그대로 상속하고 DB만 SQLite 인메모리로 교체.
      실제 MySQL 없이 테스트 가능 — Spring의 @DataJpaTest + H2 내장 DB와 동일.
"""

# 운영 설정 전체를 가져온 뒤 DB만 덮어씀
from config.settings import *  # noqa: F401, F403

# MySQL 대신 SQLite 인메모리 DB로 교체
# ":memory:" = 파일 없이 메모리에만 존재, 테스트 프로세스 종료 시 자동 소멸
DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": ":memory:",
    }
}
