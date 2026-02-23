from django.db import models


class ModelVersion(models.Model):
    """사용할 AI 모델 버전 정보를 저장하는 테이블 (model_versions)"""

    # 모델 식별 이름 (e.g. "densenet121-res224-all"), 중복 불가
    name = models.CharField(max_length=255, unique=True)
    # 모델 가중치 파일 경로 (HuggingFace 캐시 경로 또는 로컬 경로)
    weights_path = models.CharField(max_length=512)
    # 레코드 생성 시각 (자동 기록, 변경 불가)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "model_versions"  # 실제 DB 테이블명 명시

    def __str__(self):
        return self.name


class InferenceJob(models.Model):
    """추론 요청 하나하나를 추적하는 테이블 (inference_jobs)"""

    # Job 상태값 정의 (Spring의 enum과 동일, DB에는 문자열로 저장)
    class Status(models.TextChoices):
        QUEUED = "QUEUED"           # Redis 큐에 들어간 상태
        IN_PROGRESS = "IN_PROGRESS" # 워커가 처리 중인 상태
        COMPLETED = "COMPLETED"     # 추론 성공
        FAILED = "FAILED"           # 추론 실패 (재시도 초과)

    # 어떤 모델 버전으로 추론할지 (FK, 모델 삭제 시 에러로 보호)
    model = models.ForeignKey(
        ModelVersion, on_delete=models.PROTECT, related_name="jobs"
    )
    # 현재 Job 상태, 기본값은 QUEUED
    status = models.CharField(
        max_length=20, choices=Status.choices, default=Status.QUEUED
    )
    # 입력 이미지의 SHA256 해시 (중복 요청 감지용 캐시 키)
    input_sha256 = models.CharField(max_length=64, db_index=True)
    # 생성·수정 시각 (updated_at은 상태 변경 시마다 자동 갱신)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "inference_jobs"
        indexes = [
            # (status, created_at) 복합 인덱스: 상태별 정렬 조회 최적화
            models.Index(fields=["status", "created_at"], name="idx_status_created"),
        ]

    def __str__(self):
        return f"Job {self.id} [{self.status}]"


class InferenceResult(models.Model):
    """추론 결과를 저장하는 테이블 (inference_results)"""

    # job_id를 PK로 사용 (1 job = 1 result 구조 강제, Spring의 @OneToOne과 동일)
    job = models.OneToOneField(
        InferenceJob, on_delete=models.CASCADE, primary_key=True, related_name="result"
    )
    # 18개 질환별 점수 전체를 JSON으로 저장 (e.g. {"Pneumonia": 0.87, ...})
    output = models.JSONField()
    # 가장 높은 점수를 받은 질환명 (검색·분석용 인덱스 적용)
    top_label = models.CharField(max_length=100, db_index=True)
    # 결과 생성 시각
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "inference_results"

    def __str__(self):
        return f"Result for Job {self.job_id}: {self.top_label}"
