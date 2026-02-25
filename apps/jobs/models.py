from django.db import models


class ModelVersion(models.Model):
    """
    사용 중인 AI 모델 버전을 추적하는 테이블.
    현재는 단일 모델(densenet121-res224-all)만 사용하지만,
    향후 모델 교체·실험 시 버전 이력을 남기기 위해 별도 테이블로 분리했다.
    """

    # 모델 식별 이름 (e.g. "densenet121-res224-all"), unique=True로 중복 등록 방지
    name = models.CharField(max_length=255, unique=True)
    # HuggingFace 캐시 경로 또는 로컬 파일 경로 — 모델 가중치 위치 기록
    weights_path = models.CharField(max_length=512)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "model_versions"

    def __str__(self):
        return self.name


class InferenceJob(models.Model):
    """
    추론 요청 1건을 추적하는 핵심 테이블.
    비동기 처리이므로 클라이언트가 상태를 폴링할 수 있도록 status를 DB에 저장한다.
    Redis 큐는 워커에게 "처리해야 할 job_id"를 전달하는 임시 채널이고,
    실제 상태의 단일 진실 공급원(single source of truth)은 이 테이블이다.
    """

    class Status(models.TextChoices):
        QUEUED      = "QUEUED"       # 큐에 등록됨, 아직 워커가 꺼내지 않은 상태
        IN_PROGRESS = "IN_PROGRESS"  # 워커가 꺼내서 추론 중
        COMPLETED   = "COMPLETED"    # 추론 성공, InferenceResult에 결과 저장됨
        FAILED      = "FAILED"       # 3회 재시도 후에도 실패, DLQ로 이동

    # on_delete=PROTECT: 모델이 삭제되면 에러 발생 — 실수로 모델 삭제 방지
    model = models.ForeignKey(
        ModelVersion, on_delete=models.PROTECT, related_name="jobs"
    )
    status = models.CharField(
        max_length=20, choices=Status.choices, default=Status.QUEUED
    )
    # SHA256 해시: 동일 이미지 재요청 시 Redis 캐시 조회 키로 사용
    # db_index=True: input_sha256으로 단독 조회 시 빠른 검색
    input_sha256 = models.CharField(max_length=64, db_index=True)
    created_at = models.DateTimeField(auto_now_add=True)
    # auto_now=True: save() 호출마다 자동 갱신 — stuck job 감지 기준 시각으로 활용
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "inference_jobs"
        indexes = [
            # (status, created_at) 복합 인덱스를 추가한 이유:
            # /v1/ops/metrics에서 "최근 5분간 COMPLETED job 수"처럼
            # status 조건 + created_at 범위 필터를 동시에 쓰는 쿼리가 많다.
            # status 단독 인덱스보다 복합 인덱스가 이런 쿼리에 더 효율적이다.
            models.Index(fields=["status", "created_at"], name="idx_status_created"),
        ]

    def __str__(self):
        return f"Job {self.id} [{self.status}]"


class InferenceResult(models.Model):
    """
    추론 결과를 저장하는 테이블. InferenceJob과 1:1 관계.
    결과를 별도 테이블로 분리한 이유:
      - Job 상태 조회(GET /v1/jobs/{id})와 결과 조회(GET /v1/jobs/{id}/result)를
        다른 쿼리로 분리해 각각 최적화할 수 있다.
      - inference_jobs 테이블에 JSON 컬럼을 추가하면 상태 폴링 쿼리에서
        매번 큰 JSON을 읽어오는 오버헤드가 생기므로 별도 테이블이 낫다.
    """

    # job_id를 PK로 사용 = 1 Job에 1 Result만 존재 가능 (DB 수준에서 보장)
    job = models.OneToOneField(
        InferenceJob, on_delete=models.CASCADE, primary_key=True, related_name="result"
    )
    # 18개 질환별 확률 점수 전체를 JSON으로 저장 (e.g. {"Pneumonia": 0.87, ...})
    output = models.JSONField()
    # 가장 높은 점수의 질환명 — 별도 컬럼으로 추출해 인덱스 적용
    # top_label로만 필터링하는 분석 쿼리를 JSON 파싱 없이 처리 가능
    top_label = models.CharField(max_length=100, db_index=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "inference_results"

    def __str__(self):
        return f"Result for Job {self.job_id}: {self.top_label}"
