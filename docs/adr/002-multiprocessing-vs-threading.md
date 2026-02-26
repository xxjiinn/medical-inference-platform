# ADR-002: 병렬 추론 워커 — multiprocessing vs threading

## 배경

여러 추론 요청을 동시에 처리하기 위해 worker를 병렬로 실행해야 했다. Python에서 병렬 실행 방식은 크게 `threading`과 `multiprocessing` 두 가지다.

## 결정

`multiprocessing.Process`로 worker를 별도 프로세스로 실행한다.

## 이유

PyTorch 추론은 CPU-bound 작업이다. PyTorch의 C++ 연산은 GIL을 release하지만, 전처리(이미지 디코딩, numpy 변환 등 순수 Python 코드)와 제어 흐름은 GIL의 영향을 받아 직렬화된다. threading 환경에서는 전처리 단계가 병목이 된다.

`multiprocessing`은 각 worker를 독립 프로세스로 생성해 GIL 영향 없이 전처리와 추론 전 과정을 병렬 실행한다. 프로세스 격리로 한 worker 크래시가 다른 worker에 영향을 주지 않는 이점도 있다.

## 결과

- 장점: GIL 우회로 진정한 병렬 추론 가능, 한 worker 크래시가 다른 worker에 영향 없음
- 단점: 각 worker 프로세스가 모델을 독립적으로 메모리에 로드 (worker 1개당 약 500MB). worker 수를 늘릴수록 메모리 사용량이 선형 증가한다.
- t3.large(8GB RAM) 기준: worker 2개 실행 시 모델 메모리 ~1GB, 나머지 시스템 여유 충분.
