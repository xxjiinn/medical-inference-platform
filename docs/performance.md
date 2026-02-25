# 성능 분석

## 환경

- AWS EC2 t3.large (vCPU 2, RAM 8GB)
- CPU-only, Docker Compose
- 모델: torchxrayvision DenseNet121 (densenet121-res224-all)

---

## 단일 추론 (EC2 t3.large, 50회 반복)

5회 워밍업 후 50회 측정. 1회 측정은 콜드 스타트 영향을 받으므로 신뢰할 수 없다.

| p50 | p95 | p99 | mean |
|-----|-----|-----|------|
| 277ms | 304ms | 318ms | 280ms |

t3.large vCPU에서 DenseNet121 forward pass는 약 270~320ms 범위.

---

## 배치 추론 스케일링

micro-batching 설계 검증을 위해 배치 크기별 latency 측정 (각 20회 반복).

| batch size | p50 |
|-----------|-----|
| 1 | 272ms |
| 2 | 521ms |
| 4 | 1,018ms |
| 8 | 2,073ms |

CPU에서는 배치 크기에 선형 비례한다. GPU와 달리 단일 forward pass 내 병렬 처리 이득이 없다.

CPU에서 micro-batching의 실제 효과는 throughput 향상보다 **상태 전환(QUEUED→IN_PROGRESS) 쿼리 절감**에 있다. N개 job의 IN_PROGRESS 전환을 개별 UPDATE N번 대신 `filter().update()` 1번으로 처리한다. 결과 저장(InferenceResult.create + job.save)은 여전히 job별 개별 쿼리로, 전체 병목은 DB가 아닌 CPU forward pass(≈277ms)다. GPU 환경에서는 배치가 선형 이상의 throughput 이득을 주는 구조로 설계되어 있다.

---

## 부하 테스트 (Locust, 로컬 → EC2)

실제 병원 워크플로우 시뮬레이션: X-ray 업로드 → 결과 폴링 (100ms 간격).

두 가지 시나리오를 7:3 비율로 동시 실행, 각 120초 지속:
- **cache_miss**: 매 요청마다 다른 이미지 → 워커 추론 필요
- **cache_hit**: 동일 이미지 반복 제출 → SHA256 캐시 히트, 재추론 없이 즉시 반환

### 동시 사용자 10명

| 시나리오 | p50 | p95 | failure |
|---------|-----|-----|---------|
| cache_miss | 1,100ms | 2,800ms | 0% |
| cache_hit | **68ms** | 230ms | 0% |

cache_hit이 cache_miss 대비 **16배 빠름**.

### 동시 사용자 20명

| 시나리오 | p50 | p95 | failure |
|---------|-----|-----|---------|
| cache_miss | 6,000ms | 9,700ms | 0% |
| cache_hit | **110ms** | 370ms | 0% |

cache_hit이 cache_miss 대비 **54배 빠름**.

사용자 2배(10→20명) 시 cache_miss latency가 5.5배 악화됐다. Redis 큐에 job이 쌓이는 속도가 워커 처리 속도를 초과하기 때문이다. 반면 cache_hit은 재추론 없이 DB 조회만 하므로 부하와 무관하게 100ms 수준을 유지했다.

두 시나리오 모두 failure rate 0% — 부하 증가 시 latency만 증가하고 크래시 없음.

---

## 병목 분석

API 레이어(job 제출)는 20명 부하에서도 p50=92ms로 안정적. 병목은 **inference worker의 CPU 처리 속도**다. t3.large vCPU 2개 + worker 2개 구성에서 forward pass가 약 277ms이므로 이론적 최대 처리량은 약 7 RPS. 실측 처리량이 이에 미치지 못하는 이유는 전처리, DB I/O, Redis 통신 오버헤드가 더해지기 때문이다.

---

## 개선 방향

| 방법 | 예상 효과 | 비고 |
|------|-----------|------|
| GPU 인스턴스 (g4dn.xlarge) | 10~20x throughput 향상 | NVIDIA T4, GPU batching 효과 발생 |
| 수평 확장 (EC2 다중 인스턴스 + LB) | worker 수에 비례한 처리량 증가 | Redis queue 공유 구조로 바로 적용 가능 |
| ONNX Runtime | 해당 모델 적용 불가 | [ADR-003](adr/003-onnx-limitation.md) 참고 |
