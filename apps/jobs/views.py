"""
views.py
역할: HTTP 요청을 받아 비즈니스 로직 실행 후 JSON 응답 반환.
      Spring의 @RestController와 동일.
      DRF의 APIView = Spring의 @RestController + ResponseEntity 처리.
"""

import hashlib
import io

from PIL import Image
from django.db import transaction
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.throttling import AnonRateThrottle
from django.shortcuts import get_object_or_404

from .models import InferenceJob, InferenceResult, ModelVersion
from .serializers import (
    JobCreateResponseSerializer,
    JobStatusSerializer,
    InferenceResultSerializer,
)
from workers.redis_queue import enqueue, get_cache, set_cache, store_image

# 업로드 허용 최대 파일 크기: 10MB
# X-ray PNG/JPEG 변환 이미지 기준 (원본 DICOM이 아닌 클라이언트 변환본)
MAX_IMAGE_BYTES = 10 * 1024 * 1024


class JobCreateView(APIView):
    """
    POST /v1/jobs
    이미지를 받아 추론 Job을 생성하고 큐에 등록.
    Spring의 @PostMapping + @RequestPart(image) 처리와 동일.
    """

    # IP당 분당 60회 제한 — 의도치 않은 대량 요청으로 인한 GPU/CPU 자원 고갈 방지
    throttle_classes = [AnonRateThrottle]

    def post(self, request):
        # 1. 이미지 파일 추출 (multipart/form-data의 "image" 필드)
        #    Spring의 @RequestPart("image") MultipartFile과 동일
        image_file = request.FILES.get("image")
        if not image_file:
            return Response(
                {"error": "image field is required"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        # 1a. 파일 크기 검증 (읽기 전 체크 — 대용량 파일이 메모리에 풀 로드되는 것 방지)
        if image_file.size > MAX_IMAGE_BYTES:
            return Response(
                {"error": f"Image too large. Max size: {MAX_IMAGE_BYTES // 1024 // 1024}MB."},
                status=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            )

        # 1b. Content-Type 검증 — "image/"로 시작해야 함 (PDF, 실행파일 등 차단)
        content_type = getattr(image_file, "content_type", "") or ""
        if not content_type.startswith("image/"):
            return Response(
                {"error": f"Invalid content type '{content_type}'. Expected an image file."},
                status=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            )

        # 2. 이미지 bytes 읽기 + SHA256 해시 계산 (중복 감지 키)
        image_bytes = image_file.read()

        # 2a. PIL로 실제 이미지 유효성 검증 (헤더 파싱 — 손상 파일 조기 거부)
        #     verify()는 헤더만 확인하므로 전처리 실패는 워커에서 별도 처리
        try:
            Image.open(io.BytesIO(image_bytes)).verify()
        except Exception:
            return Response(
                {"error": "Invalid or corrupted image file."},
                status=status.HTTP_422_UNPROCESSABLE_ENTITY,
            )

        sha256 = hashlib.sha256(image_bytes).hexdigest()

        # 3. Redis 캐시에서 동일 이미지의 기존 job_id 조회 (중복 요청 처리)
        cached_job_id = get_cache(sha256)
        if cached_job_id:
            job = get_object_or_404(InferenceJob, pk=cached_job_id)
            # 캐시 히트 + COMPLETED: 결과를 즉시 반환 (폴링 불필요)
            if job.status == InferenceJob.Status.COMPLETED:
                result = get_object_or_404(InferenceResult, job=job)
                serializer = InferenceResultSerializer(result)
                return Response(serializer.data, status=status.HTTP_200_OK)
            # QUEUED / IN_PROGRESS: job_id 반환, 클라이언트가 폴링으로 확인
            serializer = JobCreateResponseSerializer(job)
            return Response(serializer.data, status=status.HTTP_200_OK)

        # DB fallback: Redis TTL 만료 등으로 캐시가 사라진 경우
        # FAILED를 제외한 기존 job이 DB에 있으면 재추론 없이 반환
        existing_job = (
            InferenceJob.objects.filter(input_sha256=sha256)
            .exclude(status=InferenceJob.Status.FAILED)
            .order_by("-created_at")
            .first()
        )
        if existing_job:
            set_cache(sha256, existing_job.id)  # 캐시 갱신 (다음 요청은 캐시 히트)
            if existing_job.status == InferenceJob.Status.COMPLETED:
                result = get_object_or_404(InferenceResult, job=existing_job)
                serializer = InferenceResultSerializer(result)
                return Response(serializer.data, status=status.HTTP_200_OK)
            # QUEUED / IN_PROGRESS: 이미지가 Redis에서 만료됐을 수 있으므로 재저장
            store_image(sha256, image_bytes)
            serializer = JobCreateResponseSerializer(existing_job)
            return Response(serializer.data, status=status.HTTP_200_OK)

        # [known limitation] TOCTOU race condition
        # 캐시 미스 직후 같은 이미지로 동시 요청이 오면 둘 다 job을 생성할 수 있다.
        # 드물게 중복 job이 생기더라도 마지막 set_cache()가 최신 job_id로 덮어쓰기 때문에
        # 이후 동일 이미지 요청은 정상적으로 캐시 히트를 보게 된다.
        # 완벽한 dedup을 원한다면 DB unique constraint + get_or_create가 필요하다.

        # 4. 사용할 모델 버전 조회 (DB에 등록된 최신 모델)
        #    모델이 없으면 503 반환
        model_version = ModelVersion.objects.order_by("-created_at").first()
        if not model_version:
            return Response(
                {"error": "No model version registered"},
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
            )

        # 5. DB에 새 Job 생성 (status=QUEUED)
        #    transaction.atomic()으로 DB write를 원자적으로 보장.
        #    Redis ops(6~8)는 트랜잭션 밖에서 실행 — Redis는 DB 트랜잭션에 참여 불가.
        #    만약 enqueue(7) 전 서버 크래시 시: DB에는 QUEUED job이 남지만 큐에 미등록.
        #    → _recover_stuck_jobs()의 'QUEUED stuck' 복구(5분 기준)가 자동으로 처리.
        with transaction.atomic():
            job = InferenceJob.objects.create(
                model=model_version,
                status=InferenceJob.Status.QUEUED,
                input_sha256=sha256,
            )

        # 6. 이미지 bytes를 Redis에 임시 저장 (워커가 추론 시 꺼냄, TTL 10분)
        store_image(sha256, image_bytes)

        # 7. Redis 큐에 job_id 등록 -> 워커가 꺼내서 추론 시작
        enqueue(job.id)

        # 8. SHA256 -> job_id 캐시 저장 (이후 동일 이미지 요청 시 빠르게 반환)
        set_cache(sha256, job.id)

        # 9. 생성된 Job 정보 응답 (201 Created)
        serializer = JobCreateResponseSerializer(job)
        return Response(serializer.data, status=status.HTTP_201_CREATED)


class JobStatusView(APIView):
    """
    GET /v1/jobs/{id}
    Job의 현재 상태 조회.
    Spring의 @GetMapping("/{id}") + jobRepository.findById()와 동일.
    """

    def get(self, request, job_id):
        # Job이 없으면 자동으로 404 반환 (Spring의 Optional.orElseThrow와 동일)
        job = get_object_or_404(InferenceJob, pk=job_id)
        serializer = JobStatusSerializer(job)
        return Response(serializer.data)


class JobResultView(APIView):
    """
    GET /v1/jobs/{id}/result
    추론 결과 조회. Job이 COMPLETED 상태일 때만 결과 존재.
    """

    def get(self, request, job_id):
        # Job 존재 확인
        job = get_object_or_404(InferenceJob, pk=job_id)

        # Job이 완료되지 않았으면 결과 없음을 명시
        if job.status != InferenceJob.Status.COMPLETED:
            return Response(
                {"error": f"Job is not completed yet. Current status: {job.status}"},
                status=status.HTTP_409_CONFLICT,
            )

        # InferenceResult 조회 (OneToOne 관계, job.result로 접근)
        result = get_object_or_404(InferenceResult, job=job)
        serializer = InferenceResultSerializer(result)
        return Response(serializer.data)
