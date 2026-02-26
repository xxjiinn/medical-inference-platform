# ADR-004: 프로덕션 서버 — WSGI(Gunicorn) vs ASGI(Uvicorn)

## 배경

Django의 개발용 `runserver`는 단일 스레드로 동작하며 프로덕션 환경에 적합하지 않다. 프로덕션 WSGI/ASGI 서버 선택이 필요했다.

## 결정

Gunicorn(WSGI)을 사용한다. worker 수는 t3.large vCPU 2개 기준 `2 * CPU + 1 = 5`가 권장식이나, 해당 인스턴스에서 inference worker와 리소스를 공유하는 점을 고려해 `--workers 2`로 설정했다.

## 이유

이 프로젝트의 모든 Django view는 동기 함수(`def`)다. WebSocket, Server-Sent Events, 스트리밍 응답 등 비동기 I/O가 필요한 기능이 없다. ASGI(Uvicorn)는 이런 비동기 기능을 위한 인터페이스이며, 동기 REST API에 적용하면 내부적으로 thread pool을 통해 동기 코드를 실행하는 우회 로직이 추가된다. 복잡도만 높아지고 이점이 없다.

Gunicorn은 pre-fork 모델로 각 요청을 별도 worker 프로세스가 처리한다. 동기 Django 애플리케이션에 검증된 선택이며, 설정도 단순하다.

## 결과

- 장점: 단순한 설정, 동기 Django에 최적, 프로덕션 검증된 안정성
- 단점: WebSocket 등 비동기 기능 추가 시 Uvicorn으로 전환 필요
- 향후 실시간 추론 결과 스트리밍 등이 요구된다면 ASGI 전환을 검토한다.
