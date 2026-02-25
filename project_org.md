# JLK MEDIHUB CXR Inference Serving — 프로젝트 구현 기록

> ⚠️ 이 문서는 내부 기록용. GitHub에 올리지 않음.
> 작성 목적: 면접 대비, 멘토 리뷰, 본인 복습

---

## 프로젝트 목표

JLK 백엔드 개발자 지원을 위한 포트폴리오. JLK의 실제 제품인 MEDIHUB CXR(흉부 X-ray AI 판독 시스템)의 백엔드 인프라를 직접 구현해본다. 단순 튜토리얼이 아니라 운영 가능한 수준의 시스템을 목표로 설계했다.

**핵심 기술 스택**: Django REST Framework, Redis(큐), MySQL, PyTorch(torchxrayvision DenseNet121), Docker Compose, AWS EC2, GitHub Actions CI/CD

---

## Phase 1: Docker + Django 초기 설정

### 한 것

- `docker-compose.yml` 작성: api(8000), worker, db(MySQL 8.0), redis(6380→6379) 4개 컨테이너
- Django 프로젝트 구조 설정: `config/`, `apps/jobs/`, `apps/ops/`, `workers/`
- `config/settings.py`: 환경변수로 DB/Redis 연결 설정, 필수값은 `os.environ["KEY"]`(없으면 즉시 에러)

### 에러 및 해결

**Redis 포트 충돌**

- 로컬 Mac에 Redis가 이미 6379 포트로 실행 중
- `docker-compose up` 시 `address already in use` 에러 발생
- 해결: docker-compose.yml에서 `6379:6379` → `6380:6379` 변경 (호스트 포트만 바꾸고 컨테이너 내부는 6379 유지)
- 교훈: 컨테이너 포트는 항상 호스트 포트 충돌 가능성 고려해야 함

---

## Phase 2: Django 모델 설계

### 한 것

`apps/jobs/models.py`에 3개 테이블 설계:

- `ModelVersion`: 사용 모델 버전 관리 (name UNIQUE, weights_path)
- `InferenceJob`: 추론 요청 1건 (status ENUM, input_sha256, FK→ModelVersion)
  - status: QUEUED → IN_PROGRESS → COMPLETED / FAILED
  - (status, created_at) 복합 인덱스 추가
- `InferenceResult`: 추론 결과 (job_id PK, output JSON, top_label)
  - top_label에 db_index=True

### 설계 결정

- `InferenceResult.job`을 OneToOne PK로 설계: job 1개당 result 1개 강제, 별도 join 없이 job_id로 result 조회 가능
- status를 `TextChoices`로 정의: DB에는 문자열 저장, Python에서는 `InferenceJob.Status.COMPLETED`처럼 enum으로 안전하게 참조

---

## Phase 3: 모델 로더 (model_loader.py)

### 한 것

`workers/model_loader.py`: torchxrayvision DenseNet121 싱글톤 로더

- `get_loader()`: 전역 인스턴스 반환 (프로세스당 모델 1회만 로드)
- `load()`: HuggingFace Hub에서 가중치 다운로드 또는 캐시 사용
- `preprocess()`: bytes → (1,1,224,224) PyTorch tensor 변환
- `predict()`: 단일 추론 → 18개 질환 점수 딕셔너리 반환
- `predict_batch()`: N개 텐서를 (N,1,224,224)로 묶어 배치 추론

### 에러 및 해결

**모델 다운로드 중 EOFError**

- 네트워크 불안정으로 부분 다운로드 후 파일 손상 → 워커 프로세스 크래시
- 해결: 별도 조치 없이 workers/main.py의 프로세스 모니터가 3초마다 상태 감지 → 자동 재시작 → 재다운로드 성공
- 교훈: Graceful restart 설계의 중요성. 장애 발생 시 자동 복구되는 구조

---

## Phase 4: API 엔드포인트

### 한 것

`apps/jobs/views.py`에 3개 뷰 구현:

- `JobCreateView` (POST /v1/jobs): 이미지 업로드 → SHA256 계산 → 캐시 체크 → job 생성 → Redis 이미지 저장 → 큐 등록
- `JobStatusView` (GET /v1/jobs/{id}): job 상태 반환
- `JobResultView` (GET /v1/jobs/{id}/result): 추론 결과 반환 (미완료 시 409)

### SHA256 중복 제거 설계

동일 이미지 반복 요청 시 재추론 없이 기존 결과 즉시 반환:

1. 이미지 bytes의 SHA256 해시 계산
2. Redis에서 `cache:sha256:{hash}` 키 조회
3. 캐시 히트: 기존 job_id 반환
4. 캐시 미스: 새 job 생성 + `set_cache(sha256, job.id)` 저장 (TTL 10분)

---

## Phase 5: Worker + Redis 큐

### 한 것

**`workers/redis_queue.py`**: Redis 큐 헬퍼

- `QUEUE_KEY = "inference:queue"` (LPUSH로 넣고 BRPOP으로 꺼내는 FIFO 구조)
- `enqueue()`: job_id를 큐 왼쪽에 삽입 (LPUSH)
- `collect_batch()`: BRPOP 블로킹 대기 후 30ms 윈도우로 추가 job 수집 (dequeue는 이 함수로 통합)
- `get_cache()` / `set_cache()`: SHA256 → job_id 캐시 조작
- `store_image()`: 이미지 bytes를 Redis에 임시 저장 (TTL 10분)

**`workers/worker.py`**: 실제 추론 워커

- `fetch_image_bytes()`: Redis에서 이미지 조회
- `process_batch()`: 배치 추론 핵심 함수
- `_handle_failed_jobs()`: 실패 job 재시도/DLQ 처리
- SIGALRM으로 추론 타임아웃 처리

**`workers/main.py`**: 워커 매니저

- WORKER_COUNT만큼 multiprocessing.Process로 워커 실행
- 3초마다 상태 체크, 크래시 시 자동 재시작

### 에러 및 해결

**Python stdlib 모듈명 충돌 (가장 중요한 버그)**

- `workers/queue.py` 파일을 만들었더니 redis 라이브러리 내부에서 `from queue import Empty, Full` 실행 시 stdlib `queue`가 아닌 우리 파일을 import
- 증상: `ImportError: cannot import name 'Empty' from partially initialized module 'queue'`
- 해결: `workers/queue.py` → `workers/redis_queue.py`로 파일명 변경, 관련 import 경로 전부 수정
- 교훈: Python은 현재 디렉토리 모듈이 stdlib보다 우선순위 높음. 파일명을 stdlib 모듈명과 겹치지 않게 지어야 함

### 설계 결정: Celery 미사용

Celery 대신 Redis LPUSH/BRPOP 직접 구현한 이유:

- Celery는 브로커 설정, 태스크 직렬화, Beat 스케줄러 등 추가 설정 필요
- 단순 FIFO 큐는 Redis 2개 명령으로 충분
- 오버헤드 없이 동일한 기능 → ADR-001에 문서화

### 설계 결정: multiprocessing 선택

threading 대신 multiprocessing을 선택한 이유:

- PyTorch 추론은 CPU-bound → GIL(Global Interpreter Lock)로 인해 threading은 진정한 병렬 실행 불가
- `multiprocessing.Process`는 별도 프로세스라 GIL 공유 없음 → 진정한 병렬 추론 가능
- 단점: 프로세스 간 메모리 공유 없음 → 각 워커가 모델을 독립적으로 메모리에 올림
- → ADR-002에 문서화

### 파라미터 결정 근거

**WORKER_COUNT = 2 (기본값)**
t3.large = vCPU 2개. 워커 1개 = vCPU 1개 전담 → context-switching 없이 최대 병렬 추론. 환경변수로 노출해 인스턴스 타입 변경 시 재배포 없이 조정 가능.

**BRPOP timeout = 5초**
큐가 비면 워커는 BRPOP으로 최대 5초 blocking 대기 후 `shutdown` 플래그를 확인하러 루프로 복귀. SIGTERM 수신 후 최대 5초 내 종료 보장 (graceful shutdown 응답 시간). 너무 짧으면 빈 큐에서 CPU busy-loop 발생.

**manager 점검 주기 = 3초**
이론적 최대 처리량 ≈ 7 RPS 기준, 3초 다운타임 = 약 21개 job 대기. 크래시 후 빠른 자동 재시작을 위한 최소 주기. 더 짧으면 manager 프로세스 CPU 낭비.

**INFERENCE_TIMEOUT = 10초**
p99=318ms 대비 31배 여유. "느린 추론"이 아닌 "완전히 멈춘 추론"(OOM, deadlock) 감지용. 의료 이미지는 처리 중간에 포기하면 안 되므로 보수적으로 설정. 배치에는 `10 × batch_size`초 비례 적용.

**MAX_RETRIES = 3**
일시적 장애(Redis TTL 직전 만료, 네트워크 순간 단절)는 1~2회 재시도로 해결 가능. 초과 시 DLQ로 보내 운영자가 수동 처리. 3회 이상은 DLQ 지연이 과도해지며 구조적 문제일 가능성이 높음.

**image/cache TTL = 600초 (10분)**
20명 동시 사용자 기준 p50=6,000ms. 극단적 backlog에서도 job이 10분 안에 처리됨. stuck job recovery threshold(10분)와 의도적으로 일치시킴 → 복구된 job이 image 만료로 image_not_found → retry → DLQ 경로를 자연스럽게 타도록 설계.

**retry counter TTL = 3600초 (1시간)**
image TTL(10분) × MAX_RETRIES(3) = 최대 30분의 재시도 사이클 커버. FAILED 확정 시 즉시 `r.delete()`로 삭제하므로 TTL은 프로세스 강제 종료 시 카운터 누수 방지용 안전 장치.

**stuck job threshold = 10분**
image TTL과 동일하게 설정. 10분 이상 IN_PROGRESS = 워커 크래시로 판단 (정상 추론 p99=318ms 대비 수천 배 여유). 복구 후 image도 만료되므로 image_not_found → retry → DLQ로 자연스럽게 정리됨.

---

## Phase 6: 마이크로 배칭

### 한 것

`workers/redis_queue.py`에 `collect_batch()` 추가:

1. BRPOP(blocking, 5초): 첫 번째 job 대기
2. 30ms 윈도우 동안 RPOP 반복: 추가 job 수집 (non-blocking)
3. max_size(8개) 초과 시 즉시 반환

`worker.py`의 `process_batch()`를 배치 처리로 변경:

- N개 job의 텐서를 (N,1,224,224)로 묶어 단일 forward pass

### 파라미터 결정 근거

**30ms 윈도우:**
- p50 추론 시간 = 277ms. 30ms 대기 → 30/277 ≈ **11% latency overhead**.
- "burst 요청을 묶어서 처리"라는 목표와 "latency 악화"의 균형점. 50ms면 18%, 100ms면 36% 오버헤드로 CPU 배칭 이득 없는 환경에서 비합리적.
- `BATCH_WINDOW_MS` 환경변수로 설정 가능 — 실제 트래픽 패턴에 맞게 운영자가 조정 가능하도록 설계.

**max_size=8:**
- CPU 선형 스케일링 기준: bs=8 → 8×277ms ≈ **2,200ms** 최대 배치 처리 시간.
- 10명 동시 사용자 p95 목표 ≈ 2,800ms. 배치 하나의 처리 시간이 SLA 내에 들어오는 최대치.
- bs=16이면 ≈4,400ms — p95 목표 초과.
- 예상 트래픽 5–20 RPS에서 30ms 윈도우로 수집되는 job은 평균 0.15–0.6개 → 대부분 배치 크기 1–2. max_size=8은 예상치 못한 burst 트래픽에 대한 안전 상한선.

### 설계 의도 vs 실제 결과

CPU에서는 배치 크기에 선형 비례 (bs=1: 272ms, bs=8: 2073ms ≈ 8배). GPU와 달리 배치로 인한 처리량 향상 없음. EC2 벤치마크로 성능 향상 없음 확인. micro-batching의 실제 효과는 상태 전환 쿼리 절감(QUEUED→IN_PROGRESS를 `filter().update()` 1회로 처리). 결과 저장은 job별 개별 쿼리이며 전체 병목은 CPU forward pass(≈277ms). GPU 환경에서는 배치 추론이 선형 이상의 처리량 이득을 준다.

### 에러 및 해결

**테스트에서 StopIteration**

- collect_batch에서 rpop None 수신 후 재시도 루프였는데, 테스트 mock의 side_effect 리스트가 고갈
- 해결: None 수신 즉시 break로 변경. 빈 큐면 추가 수집 의미 없음
- 교훈: sleep+retry 루프는 테스트에서 예측 불가한 호출 횟수 발생

---

## Phase 7: ONNX 최적화 시도

### 한 것

추론 속도 개선 목적으로 ONNX Runtime 적용 시도:

- `scripts/convert_to_onnx.py`: 모델을 ONNX 포맷으로 변환
- `workers/onnx_loader.py`: ONNX Runtime 기반 추론 로더 (ModelLoader와 동일한 인터페이스)
- `scripts/benchmark.py`: PyTorch vs ONNX 성능 비교

### 에러 발생

변환은 성공했으나 실제 추론 시 에러:

```
Non-zero status code returned while running Reshape node /Reshape_2.
Input shape:{11}, requested shape:{}
```

### 근본 원인 분석 과정

단순 에러 메시지가 아닌 ONNX 그래프를 직접 분석:

- `onnx` 라이브러리로 Reshape_2 노드 주변 연산 순서 확인
- `And → NonZero → Transpose → GatherND → Div → Reshape_2`
- `NonZero` 연산: 입력 텐서에서 0이 아닌 원소의 인덱스 반환 → 출력 크기가 런타임 입력값에 따라 달라짐 (data-dependent output shape)
- export 시 zero 더미 입력 → NonZero가 0개 반환 → 스칼라([])로 고정
- 실제 추론 시 NonZero가 11개 반환 → reshape([11], []) = 불가
- batch_size=2면 {22}, batch=4면 {44}: 배치 크기에 정비례하는 것도 확인

### 근본 원인

`densenet121-res224-all`은 NIH, PadChest, CheXpert 등 7개 데이터셋 앙상블 모델. 각 병리별로 "어떤 데이터셋이 이 병리 레이블을 포함하는가"를 NonZero로 동적 검색한 뒤 GatherND로 집계하는 구조. ONNX는 정적 계산 그래프(static computation graph)만 지원 → 이 data-dependent control flow 표현 불가.

### 결론

ONNX 적용 불가 (모델 아키텍처 근본 제약, 우회 방법 없음). ADR-003에 문서화. onnx_loader.py는 코드 산출물로 보존 — 단일 데이터셋 모델로 교체 시 재활용 가능.

---

## Phase 8: 운영 지표 엔드포인트

### 한 것

`apps/ops/views.py`:

- `MetricsView` (GET /v1/ops/metrics): 최근 5분간 처리량(RPS), 실패율, latency p50/p95/p99
  - latency: `InferenceResult.created_at - InferenceJob.created_at` → Django ORM annotate + ExpressionWrapper
  - numpy로 백분위수 계산
- `DLQView` (GET /v1/ops/dlq): Redis DLQ에서 최종 실패 job 목록 조회

### DLQ(Dead Letter Queue) 설계

재시도 3회 초과 후 영구 실패한 job 처리:

- `workers/redis_queue.py`에 `DLQ_KEY = "dlq:failed_jobs"` 추가
- `_handle_failed_jobs()`에서 MAX_RETRIES 초과 시 `r.lpush(DLQ_KEY, job.id)` 저장
- 운영자가 `/v1/ops/dlq`로 조회 후 원인 파악 및 수동 재처리 가능

---

## Phase 9: 테스트

### 한 것

pytest-django 기반 테스트 (총 26개):

- `tests/conftest.py`: 공통 fixtures (SQLite in-memory, api_client, sample_image, model_version)
- `tests/test_views.py`: API 엔드포인트 (JobCreate/Status/Result)
- `tests/test_queue.py`: collect_batch, enqueue, get/set_cache (Mock Redis)
- `tests/test_metrics.py`: MetricsView (집계 로직, 시간 윈도우)
- `tests/test_worker.py`: process_batch, \_handle_failed_jobs (Mock 모델+Redis)

### 테스트 전략

- 실제 Redis/모델 대신 Mock 사용: 테스트 속도 (4초 이내), CI 환경 의존성 제거
- DB는 SQLite in-memory로 교체: MySQL 없이 비즈니스 로직만 검증
- 통합 테스트(전체 스택 연동)는 EC2 배포 후 실시: curl로 초기 smoke test, Locust로 부하 테스트

**Locust가 통합 테스트 역할을 하는 이유**: `_submit_and_wait()`이 status=COMPLETED를 확인하기 위해 실제 경로 전체를 거침:
1. Redis (image 저장 + enqueue) → 2. Worker BRPOP + 실제 PyTorch 추론 → 3. MySQL (InferenceResult + status 업데이트) → 4. API MySQL 조회 → COMPLETED 반환.
COMPLETED 응답 자체가 Redis+MySQL+Worker 전부 정상 작동 증거.

**단위 테스트가 커버 못 하는 경로**: retry 카운터 Redis 연산, DLQ push, stuck job 복구 — Mock으로만 검증. 실제 Redis 연동 후 이 경로들은 수동 확인하지 않음. (면접 시 솔직히 인정할 부분)

### 에러 및 해결

**pytest가 실제 MySQL을 쓰는 문제**

- `with override_settings(DATABASES=SQLite):` 사용했으나 yield 없이 with 블록 종료
- 결과: 테스트 실행 시점에는 MySQL 사용 → `Duplicate entry 'densenet121-res224-all'` 에러
- 해결: `config/test_settings.py` 별도 생성 + `pytest.ini`의 `DJANGO_SETTINGS_MODULE = config.test_settings`
- 교훈: Django override_settings는 런타임 패치. DB 격리는 settings 파일 레벨에서 처리해야 함

**DRF 파일 업로드 형식 오류**

- `{"image": ("file.png", bytes, "image/png")}` → requests 라이브러리 전용 형식
- DRF APIClient는 file-like object 필요 → `request.FILES.get("image")` = None → 400 에러
- 해결: `io.BytesIO(bytes)` + `.name = "test.png"` 속성 추가

**auto_now_add=True 우회 문제**

- 테스트에서 `create(created_at=과거시간)` 전달해도 Django가 무시 → 전부 현재 시간으로 저장
- 증상: 5분 윈도우 테스트에서 10분 전 데이터도 집계됨
- 해결: `create()` 후 `QuerySet.filter(pk=...).update(created_at=과거시간)` 우회
- 교훈: auto_now_add는 create()에서 값 지정 불가. update()는 이 제약을 우회함

---

## Phase 10: 배포 (EC2 + CI/CD)

### 한 것

**Docker 설정 개선**

- `scripts/entrypoint.sh`: `python manage.py runserver` → `gunicorn config.wsgi:application --bind 0.0.0.0:8000 --workers 2`
- `config/wsgi.py` 신규 생성 (누락됐었음)
- `requirements.txt`에 `gunicorn==21.*` 추가

**AWS EC2 배포**

- t3.large (vCPU 2, RAM 8GB), Ubuntu 22.04 LTS
- `/home/ubuntu/medical-inference-platform/`에 저장소 clone
- `.env` 파일 수동 생성 (DB 비밀번호, SECRET_KEY 등)
- Docker 4개 컨테이너 모두 실행 확인

**GitHub Actions CI/CD**

- `.github/workflows/deploy.yml`: push → 테스트 → EC2 자동 배포
- test job: pytest (Redis 서비스 컨테이너 포함)
- deploy job: SSH로 EC2 접속 → git pull → docker compose up --build

### 에러 및 해결

**gunicorn 실행 실패**

- `gunicorn.errors.HaltServer` 에러로 api 컨테이너 즉시 종료
- 원인: `config/wsgi.py` 파일 자체가 없었음
- 해결: `config/wsgi.py` 신규 생성

**docker-compose v1 KeyError**

- EC2에 구버전 docker-compose 설치 → `KeyError: 'ContainerConfig'` 에러
- 해결: Docker Compose v2 플러그인을 바이너리로 직접 다운로드해 `/usr/local/lib/docker/cli-plugins/`에 설치

**컨테이너 이름 충돌**

- 기존 컨테이너가 남아있어 새로 올릴 때 이름 충돌
- 해결: `docker compose down && docker compose up -d --build`

**CI에서 KeyError: 'MYSQL_DATABASE'**

- GitHub Actions 테스트 환경에서 환경변수 없어서 `settings.py` import 시 KeyError
- 해결: workflow의 test step에 dummy MySQL 환경변수 추가 (`MYSQL_DATABASE=test`, 등)

**GitHub PAT 권한 부족 (workflow 파일 push 거부)**

- `refusing to allow... without workflow scope` 에러
- 해결: GitHub → Settings → Developer settings → PAT에서 `workflow` 스코프 추가

**EC2에서 docker compose 명령어 없음**

- `docker compose` (v2)가 인식 안 됨
- 원인: PATH에 `/usr/local/lib/docker/cli-plugins/` 미포함
- 해결: 바이너리 직접 설치 경로 확인 후 재시도

**EC2 Locust 아키텍처 불일치**

- EC2(aarch64)에서 로컬(x86_64)용 Locust 패키지 설치 시도 → `incompatible architecture` 에러
- 해결: EC2에서 별도 가상환경(`/tmp/locust_env`) 생성 후 설치

**`.env` 파일의 선언되지 않은 변수 경고**

- `${mq9w}` 변수 경고 표시
- 원인: EC2 `.env` 파일에 실수로 입력된 변수명
- 결과: 실제 동작에는 영향 없음 (경고만 출력됨)

**REDIS_URL 환경변수 처리 방식 불일치**

- `redis_queue.py`에서 `os.environ.get("REDIS_URL", "redis://localhost:6379/0")` 기본값 방식 사용
- `settings.py`는 `os.environ["REDIS_URL"]` 필수값 방식
- 문제: 로컬에서 REDIS_URL 없이 실행 시 워커는 localhost로 연결, API는 에러 → 불일치
- 해결: `redis_queue.py`도 `os.environ["REDIS_URL"]`로 통일

---

## Phase 11: 구조화 로그 + DLQ + 테스트 보강

### 한 것

**구조화 로그 (`workers/worker.py`)**

- `log(event, **kwargs)`: JSON 형식으로 출력하는 분석용 로그 (컴퓨터가 읽기 좋은 형식)
  ```python
  {"event": "inference_completed", "job_id": 3, "top_label": "Effusion", "latency_ms": 287.4}
  ```
- `logger.*()`: 이모지 + 한글 포함 운영자가 읽기 좋은 로그
- 주요 이벤트: batch_start, image_not_found, preprocess_failed, inference_timeout, inference_error, inference_completed, job_retry, job_failed

**DLQ (Dead Letter Queue)**

- 재시도 3회 초과 job을 Redis `dlq:failed_jobs` 리스트에 저장
- `GET /v1/ops/dlq`: 해당 job들의 DB 정보 반환
- 운영자가 장애 원인 파악 및 수동 재처리에 사용

**테스트 보강 (`tests/test_worker.py` 신규)**

- `process_batch()` 정상 경로: COMPLETED + InferenceResult 생성 확인
- `process_batch()` 이미지 없음: 재시도 큐 등록 확인
- `process_batch()` 전처리 실패: InferenceResult 없음 확인
- `process_batch()` 없는 job_id: 예외 없이 스킵 확인
- `_handle_failed_jobs()` 재시도 경로: LPUSH 재등록 확인
- `_handle_failed_jobs()` DLQ 경로: FAILED 상태 + DLQ push + retry 카운터 삭제 확인

---

## Phase 12: 전체 감사 및 수정

완성 후 면접 합격 관점에서 전체 코드/문서를 재점검.

### 발견된 문제들과 처리

**1. README 성능 수치 불일치 (Critical)**

- README의 "동시 10명 p50=2,822ms"가 docs/performance.md의 "p50=1,300ms"와 불일치
- 원인: 서로 다른 측정 시점의 수치를 혼용
- 해결: EC2에서 재측정 후 통일 (아래 Phase 13)

**2. 단일 요청 p50/p95/p99가 모두 655ms (Critical)**

- 1회 측정을 3개 백분위수로 표기 → 통계적으로 의미 없음
- 해결: EC2에서 50회 반복 측정으로 교체

**3. locustfile의 FAILED job 성공 처리 버그**

- FAILED 상태도 break만 하고 `response.failure()` 미호출 → Locust 보고서에 성공으로 집계
- 해결: `response.failure()` 호출 추가

**4. Locust 측정 방법론 문제**

- `time.sleep(0.5)` 폴링 → latency 측정 오차 최대 500ms
- Locust 내장 response_time = 개별 HTTP 요청 시간 (수십 ms) ≠ 추론 latency
- 해결: 폴링을 0.1s로 줄이고, 제출부터 COMPLETED까지 end-to-end 시간을 Locust 커스텀 이벤트로 직접 보고

**5. SHA256 캐시 효과 미측정**

- 핵심 설계 결정으로 내세우는데 실제 부하 테스트에서는 매번 랜덤 이미지 사용 → 캐시 히트율 0%
- 해결: locustfile에 고정 이미지 task 추가 (cache_hit 30% : cache_miss 70%)
- `on_start()`에서 고정 이미지를 미리 처리(warmup)해 캐시 준비

**6. stuck IN_PROGRESS job 처리 없음**

- 워커가 SIGKILL 등으로 강제 종료 시 IN_PROGRESS 상태 job이 영원히 방치
- 해결: `main.py`에 `_recover_stuck_jobs()` 추가
  - 10분마다 `updated_at < now - 10분`인 IN_PROGRESS job을 QUEUED로 되돌려 재큐잉
  - 참고: image TTL도 10분이므로 복구된 job은 image_not_found → 재시도 → DLQ 경로로 흐름

**7. torch.set_num_threads(1) 제거**

- 이전에 "2 workers × multi-thread = vCPU 경합" 가설로 적용
- 실측 결과: p50이 4.9s → 5.1s로 오히려 약간 악화 (PyTorch Conv2d가 이미 내부적으로 thread 효율 관리)
- 해결: 코드에서 제거 + performance.md에서 해당 섹션 삭제

**8. benchmark.py 배치 측정 크래시**

- ONNX 모델 파일 없을 때 `onnx_loader.load()`에서 FileNotFoundError → 배치 측정 전체 중단
- 해결: try/except로 감싸 ONNX 없을 때 PyTorch만 측정하도록 변경

**9. ADR-003 일관성 문제**

- "ONNX 적용하지 않는다"고 했는데 `workers/onnx_loader.py` 137줄이 존재 → 혼란 가능성
- 해결: ADR-003에 "onnx_loader.py는 코드 산출물로 보존, 단일 데이터셋 모델 교체 시 재활용 가능" 1줄 추가

**10. README 테스트 수 오류**

- "18개 테스트"라고 적혀있는데 test_worker.py 6개 추가로 실제 24개
- 해결: 24개로 수정 (이후 recovery 테스트 2개 추가 → 최종 26개)

**11. CLAUDE.md "ADR-001 through ADR-005"**

- 실제로는 ADR 4개뿐 (005 없음)
- 해결: "ADR-001 through ADR-004"로 수정

**12. stuck job 무한 루프 가능성 (코드 버그)**

- `_recover_stuck_jobs()`가 QUEUED로 되돌리기만 하고 retry counter를 증가시키지 않음
- mid-inference SIGKILL 시 `_handle_failed_jobs()`가 실행 안 돼 retry counter 미증가
- 결과: 워커가 매번 인퍼런스 중 크래시하면 10분마다 무한 재큐잉
- 해결: `_recover_stuck_jobs()`에도 `r.incr(retry_key)` 추가 — recovery 시도 자체를 재시도 횟수에 포함. MAX_RETRIES 초과 시 FAILED + DLQ로 처리
- 테스트 추가: `test_recover_stuck_jobs_requeue`, `test_recover_stuck_jobs_dlq_on_max_retries` (총 26개)

**13. 실제 X-ray sanity check 미실시**

- 모든 테스트에 더미 이미지(완전 검정, 단색, 랜덤 노이즈)만 사용
- `scripts/validate_model.py` 작성 + torchxrayvision 테스트 슈트의 NIH 샘플 X-ray 실행
- 결과: top_label=Cardiomegaly(0.62), 18개 병리 다른 점수 분포 확인 → 전처리 파이프라인 및 모델 출력 정상
- 교훈: 인프라 구현 전 모델 출력 sanity check를 먼저 했어야 함

---

## Phase 13: EC2 성능 측정

### benchmark.py (EC2, 50회 반복)

| 지표 | 수치  |
| ---- | ----- |
| p50  | 277ms |
| p95  | 304ms |
| p99  | 318ms |
| mean | 280ms |
| min  | 264ms |
| max  | 322ms |

이전 "655ms"는 콜드 스타트 1회 측정이었음. 워밍업 5회 + 50회 반복 측정이 신뢰할 수 있는 수치.

**배치 스케일링:**
| batch size | p50 |
|-----------|-----|
| 1 | 272ms |
| 2 | 521ms |
| 4 | 1,018ms |
| 8 | 2,073ms |

CPU에서 선형 증가 → micro-batching의 throughput 효과 없음. GPU 환경에서는 다를 것.

### Locust 부하 테스트 (로컬 → EC2)

**동시 10명 (120초):**
| 시나리오 | p50 | p95 | failure |
|---------|-----|-----|---------|
| cache_miss | 1,100ms | 2,800ms | 0% |
| cache_hit | 68ms | 230ms | 0% |

cache_hit이 **16배 빠름**.

**동시 20명 (120초):**
| 시나리오 | p50 | p95 | failure |
|---------|-----|-----|---------|
| cache_miss | 6,000ms | 9,700ms | 0% |
| cache_hit | 110ms | 370ms | 0% |

cache_hit이 **54배 빠름**.

사용자 2배 시 cache_miss가 5.5배 악화된 이유: Redis 큐에 job 누적 속도가 worker 처리 속도 초과.
cache_hit은 재추론 없이 DB 조회만 → 부하와 무관하게 안정적.

---

## 최초 locustfile 문제 (측정 실패 사례)

처음 로컬 Docker(`localhost:8000`)를 대상으로 테스트 실행했을 때:

- `e2e/cache_hit` 97.67% 실패 (`status=QUEUED`)
- 원인: 10명이 동시에 시작하면서 고정 이미지 job이 아직 QUEUED 상태인데 다른 유저들이 cache_hit 요청 → 같은 job_id를 받았지만 15초 내 COMPLETED 안 됨
- 이후 20명 테스트에서는 고정 이미지가 이미 COMPLETED → cache_hit p50=14ms
- 깨달음: SHA256 캐시는 "이미 완료된 job을 재사용"하는 설계. "진행 중인 job의 중복 요청"에는 효과 없음.
- 해결: `on_start()`에서 고정 이미지를 미리 처리 완료 후 테스트 시작

---

## 미완료 사항 및 알려진 한계

### 코드 측면

1. **stuck job 복구 후 이미지 없음 문제**
   - IN_PROGRESS stuck job을 QUEUED로 복구해도, image TTL(10분)이 이미 만료된 경우 worker가 image_not_found → retry → DLQ 경로로 처리됨
   - 완전한 해결은 이미지를 Redis 대신 S3에 저장하는 것 (현재는 Redis TTL 의존)

2. **통합 테스트 없음**
   - pytest는 Mock 기반 단위 테스트만 존재
   - 실제 Redis + 실제 모델을 사용하는 pytest 통합 테스트는 EC2 배포 + curl + Locust로 대체

### 성능 측면

1. **CPU-only 한계**: t3.large에서 DenseNet121 forward pass 약 277ms → 이론 최대 처리량 ~7 RPS
2. **micro-batching 효과 없음**: CPU 선형 스케일링으로 throughput 이득 없음 (GPU에서는 효과 발생 예정)
3. **시도했으나 실패한 최적화**:
   - ONNX: 모델 아키텍처 근본 제약으로 불가 (NonZero+GatherND)
   - torch.set_num_threads(1): 오히려 약간 악화
4. **미시도 최적화**: INT8 양자화, TorchScript → accuracy 검증 환경 부재로 범위 밖으로 결정

### 의료 도메인 측면

1. **DICOM 미지원**: 실제 병원 PACS 시스템은 DICOM 포맷 사용. 현재 PNG/JPEG만 처리.
2. **PACS 연동 없음**: C-STORE, DICOMweb 등 실제 PACS 프로토콜 미구현
3. **실제 X-ray 이미지 sanity check**: `scripts/validate_model.py`로 NIH ChestX-ray14 샘플(torchxrayvision 테스트 슈트의 `00000001_000.png`)을 실행해 모델 출력 확인.
   - 전처리: PNG(0~255) → `xrv.utils.normalize(maxval=255)` → [-1024.0, 931.3] (정상 범위)
   - 추론 결과: 18개 병리 모두 다른 점수, top_label=Cardiomegaly(0.62) — 더미 이미지와 달리 해부학적 구조를 반영한 의미있는 분포 확인
   - 모델 정확도 검증(ground truth 비교)은 포트폴리오 범위 밖으로 결정. 이 테스트는 "전처리 파이프라인이 올바르고 모델이 실제 X-ray에 대해 비자명한 출력을 내는가"를 확인하는 sanity check.
4. 포트폴리오 범위는 "inference serving 인프라"로 제한

---

## 전체 파일 구조 요약

```
ai-serving/
├── apps/
│   ├── jobs/           # 추론 job API (views, models, serializers, urls)
│   └── ops/            # 운영 지표 API (MetricsView, DLQView)
├── config/             # Django 설정 (settings, urls, wsgi, test_settings)
├── docs/
│   ├── adr/            # Architecture Decision Records (001~004)
│   └── performance.md  # 성능 측정 결과
├── scripts/
│   ├── benchmark.py        # PyTorch 단일/배치 추론 성능 측정
│   ├── convert_to_onnx.py  # ONNX 변환 시도 (결과: 실패 — ADR-003)
│   ├── locustfile.py       # 부하 테스트 (cache_miss/cache_hit 시나리오)
│   ├── seed_model.py       # DB에 ModelVersion 레코드 생성
│   ├── validate_model.py   # 실제 X-ray 이미지로 모델 출력 sanity check
│   └── warmup.py           # 서버 시작 후 모델 워밍업
├── tests/              # pytest 테스트 (26개)
├── workers/
│   ├── main.py         # 워커 매니저 (프로세스 풀 관리, stuck job 복구)
│   ├── model_loader.py # PyTorch 모델 싱글톤
│   ├── onnx_loader.py  # ONNX 로더 (현재 미사용, 코드 산출물로 보존)
│   ├── redis_queue.py  # Redis 큐/캐시 헬퍼
│   └── worker.py       # 배치 추론 워커
├── docker-compose.yml
├── Dockerfile
├── Makefile
└── .github/workflows/deploy.yml  # CI/CD
```

---

## 면접 핵심 질문 대비

**"Celery 대신 Redis 직접 구현한 이유?"**
단순 FIFO 큐에 Celery의 브로커 설정, 태스크 직렬화 오버헤드가 불필요. LPUSH/BRPOP 2개 명령으로 동일 기능 구현.

**"multiprocessing 선택 이유?"**
PyTorch 추론은 CPU-bound. threading은 GIL로 진정한 병렬 실행 불가. 각 worker를 별도 프로세스로 실행해 GIL 우회.

**"ONNX 왜 안 됐나?"**
densenet121-res224-all은 7개 데이터셋 앙상블 구조. 각 병리별로 NonZero+GatherND로 동적 인덱싱. ONNX는 정적 그래프만 지원 → data-dependent output shape 표현 불가. ONNX 그래프 직접 분석해서 Reshape_2 노드에서 확인.

**"GPU 왜 안 씀?"**
개인 포트폴리오 비용 문제. 아키텍처는 GPU 전환 고려해 설계 (INFERENCE_ENGINE 환경변수, worker 분리).

**"성능 최적화는?"**
캐시 히트 16배 차이(68ms vs 1100ms) 실측. ONNX 실패 원인 규명. torch threads 실험 → 약화 → 제거. 병목은 CPU forward pass → GPU 또는 수평 확장이 다음 단계.

**"양자화나 TorchScript는 왜 안 했나?"**
densenet121-res224-all은 ONNX 분석 결과 dynamic computation 구조. TorchScript도 동일 문제 예상. INT8 양자화는 accuracy 영향 검증 환경 미비로 범위 밖으로 결정.

**"워커가 갑자기 죽으면?"**
세 가지 보호 장치:

1. main.py가 3초마다 상태 체크 → 크래시 감지 시 새 프로세스로 자동 재시작
2. 10분마다 IN_PROGRESS stuck job 복구 → QUEUED로 되돌려 재큐잉 + retry 카운터 증가 (MAX_RETRIES 초과 시 FAILED+DLQ)
3. 5분마다 QUEUED stuck job 복구 → API 서버 크래시로 enqueue 유실된 job 재등록 (image TTL=10분이므로 5분 기준 복구 시 이미지 유효)
   단, IN_PROGRESS 복구 후 image TTL(10분)이 이미 만료된 경우 image_not_found → retry → DLQ 경로로 흐름. 완전한 해결은 Redis 대신 S3에 이미지 저장.
