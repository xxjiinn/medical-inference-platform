"""
views.py
역할: HTTP 요청을 받아 비즈니스 로직 실행 후 JSON 응답 반환.
      Spring의 @RestController와 동일.
      DRF의 APIView = Spring의 @RestController + ResponseEntity 처리.
"""

import hashlib

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.shortcuts import get_object_or_404

from .models import InferenceJob, InferenceResult, ModelVersion
from .serializers import (
    JobCreateResponseSerializer,
    JobStatusSerializer,
    InferenceResultSerializer,
)
from workers.redis_queue import enqueue, get_cache, set_cache, store_image


class JobCreateView(APIView):
    """
    POST /v1/jobs
    이미지를 받아 추론 Job을 생성하고 큐에 등록.
    Spring의 @PostMapping + @RequestPart(image) 처리와 동일.
    """

    def post(self, request):
        # 1. 이미지 파일 추출 (multipart/form-data의 "image" 필드)
        #    Spring의 @RequestPart("image") MultipartFile과 동일
        image_file = request.FILES.get("image")
        if not image_file:
            return Response(
                {"error": "image field is required"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        # 2. 이미지 bytes 읽기 + SHA256 해시 계산 (중복 감지 키)
        image_bytes = image_file.read()
        sha256 = hashlib.sha256(image_bytes).hexdigest()

        # 3. Redis 캐시에서 동일 이미지의 기존 job_id 조회 (중복 요청 처리)
        cached_job_id = get_cache(sha256)
        if cached_job_id:
            # 캐시 히트: DB에서 기존 Job을 찾아 바로 반환 (재추론 생략)
            job = get_object_or_404(InferenceJob, pk=cached_job_id)
            serializer = JobCreateResponseSerializer(job)
            return Response(serializer.data, status=status.HTTP_200_OK)

        # 4. 사용할 모델 버전 조회 (DB에 등록된 최신 모델)
        #    모델이 없으면 503 반환
        model_version = ModelVersion.objects.order_by("-created_at").first()
        if not model_version:
            return Response(
                {"error": "No model version registered"},
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
            )

        # 5. DB에 새 Job 생성 (status=QUEUED)
        #    Spring의 jobRepository.save(new Job(...))와 동일
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
