# ADR-001: 태스크 큐 — Redis 직접 구현 vs Celery

## 배경

비동기 추론 Job을 처리하기 위해 태스크 큐가 필요했다. Python 생태계에서 가장 널리 쓰이는 선택지는 Celery였고, Redis를 직접 사용하는 방식도 고려했다.

## 결정

Celery를 사용하지 않고 Redis의 `LPUSH` / `BRPOP` 명령어로 FIFO 큐를 직접 구현했다.

## 이유

메시지 큐의 핵심 동작 원리를 직접 구현하며 내부 구조를 학습하고자 Celery 대신 Redis LPUSH/BRPOP을 직접 사용하기로 했다.

Celery는 브로커 설정, 태스크 등록, 직렬화/역직렬화 레이어가 추가되어 단일 태스크 유형(X-ray 추론)만 처리하는 이 구조에서는 학습 범위를 넘어선 추가 복잡성이 된다. 실무에서는 Celery가 검증된 표준 선택이다.

Redis `LPUSH` / `BRPOP` 조합은 FIFO를 보장하며, BRPOP의 blocking 특성 덕분에 worker가 idle 상태에서도 불필요한 CPU를 쓰지 않는다. 재시도 카운터는 Redis `INCR`로 관리(`retry:{job_id}`, TTL 1시간)해 DB 스키마 변경 없이 구현했다.

## 결과

- 장점: 의존성 최소화, Docker 서비스 추가 없음, 코드 단순화
- Celery 미사용 시 포기하는 기능을 직접 구현: Retry 최대 3회(`retry:{job_id}` Redis INCR), Dead Letter Queue(`dlq:failed_jobs`), 운영 모니터링(`/v1/ops/metrics`, `/v1/ops/dlq`)
- 단점: 태스크 우선순위 큐, 태스크 단위 실행 이력 조회 등은 미구현
- 확장 시: 트래픽이 늘어 Celery 수준의 기능이 필요해지는 시점에 전환 검토 가능. 현재 구조에서 Redis는 그대로 브로커로 재사용 가능하다.
