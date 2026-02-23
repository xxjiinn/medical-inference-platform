#!/bin/bash
# entrypoint.sh
# 역할: API 컨테이너 시작 시 순서대로 초기화 작업 수행 후 서버 실행.
#       Spring Boot의 application.yml spring.sql.init + Flyway migrate 자동 실행과 동일.

set -e  # 어느 명령이든 실패하면 즉시 중단

echo "[Entrypoint] Running database migrations..."
# Django migration: Spring의 Flyway migrate와 동일
python manage.py migrate --noinput

echo "[Entrypoint] Seeding initial model version..."
# ModelVersion 레코드가 없으면 생성 (멱등성 보장)
python manage.py seed_model

echo "[Entrypoint] Starting Gunicorn server..."
# 2 workers: t3.large 2 vCPU 기준 (권장: 2 * CPU + 1)
# --bind 0.0.0.0:8000: 컨테이너 외부에서 접근 가능하도록
exec gunicorn config.wsgi:application --bind 0.0.0.0:8000 --workers 2
