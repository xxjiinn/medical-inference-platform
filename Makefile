# Makefile
# 역할: 자주 쓰는 docker-compose 명령어를 짧게 줄여서 사용.

.PHONY: up down logs test benchmark convert-onnx migrate shell

# 모든 컨테이너 백그라운드 실행
up:
	docker-compose up -d

# 모든 컨테이너 중지 및 제거
down:
	docker-compose down

# 전체 로그 스트리밍
logs:
	docker-compose logs -f

# API 컨테이너 로그만
logs-api:
	docker-compose logs -f api

# Worker 컨테이너 로그만
logs-worker:
	docker-compose logs -f worker

# DB migration 실행
migrate:
	docker-compose exec api python manage.py migrate

# 단위 테스트 실행
test:
	docker-compose exec api pytest tests/ -v

# PyTorch vs ONNX 벤치마크 실행
benchmark:
	docker-compose exec worker python scripts/benchmark.py

# ONNX 변환 실행 (1회만)
convert-onnx:
	docker-compose exec worker python scripts/convert_to_onnx.py

# Django 쉘 접속
shell:
	docker-compose exec api python manage.py shell
