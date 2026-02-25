# ADR-002: 병렬 추론 워커 — multiprocessing vs threading

- **상태**: 결정됨
- **날짜**: 2026-02

## 배경

여러 추론 요청을 동시에 처리하기 위해 worker를 병렬로 실행해야 했다. Python에서 병렬 실행 방식은 크게 `threading`과 `multiprocessing` 두 가지다.

## 결정

`multiprocessing.Process`로 worker를 별도 프로세스로 실행한다.

## 이유

PyTorch 추론은 CPU-bound 작업이다. Python의 GIL(Global Interpreter Lock)은 동일 프로세스 내에서 한 번에 하나의 thread만 Python 바이트코드를 실행할 수 있도록 제한한다. threading을 사용하면 여러 thread가 존재하더라도 추론 연산이 실질적으로 직렬화된다.

`multiprocessing`은 각 worker를 독립 프로세스로 생성해 GIL의 영향을 받지 않는다. 각 프로세스는 자체 메모리 공간을 가지므로 PyTorch의 멀티프로세싱 관련 충돌도 없다.

## 결과

- 장점: GIL 우회로 진정한 병렬 추론 가능, 한 worker 크래시가 다른 worker에 영향 없음
- 단점: 각 worker 프로세스가 모델을 독립적으로 메모리에 로드 (worker 1개당 약 500MB). worker 수를 늘릴수록 메모리 사용량이 선형 증가한다.
- t3.large(8GB RAM) 기준: worker 2개 실행 시 모델 메모리 ~1GB, 나머지 시스템 여유 충분.
