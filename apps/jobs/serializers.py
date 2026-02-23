"""
serializers.py
역할: API 요청/응답 데이터 구조 정의 및 직렬화(JSON 변환).
      Spring의 DTO(Data Transfer Object) 클래스와 동일한 역할.
      ModelSerializer = JPA Entity를 기반으로 DTO를 자동 생성하는 것과 유사.
"""

from rest_framework import serializers
from .models import InferenceJob, InferenceResult


class JobCreateResponseSerializer(serializers.ModelSerializer):
    """
    POST /v1/jobs 응답 DTO.
    Job 생성 직후 반환: id와 status만 포함.
    """
    class Meta:
        model = InferenceJob
        fields = ["id", "status", "created_at"]  # 클라이언트에 노출할 필드만 선택


class JobStatusSerializer(serializers.ModelSerializer):
    """
    GET /v1/jobs/{id} 응답 DTO.
    Job의 현재 상태와 시간 정보 반환.
    """
    class Meta:
        model = InferenceJob
        fields = ["id", "status", "created_at", "updated_at"]


class InferenceResultSerializer(serializers.ModelSerializer):
    """
    GET /v1/jobs/{id}/result 응답 DTO.
    추론 결과(18개 질환 점수 + top_label) 반환.
    """
    # job_id를 명시적으로 포함 (OneToOne PK라 자동 노출이 안 됨)
    job_id = serializers.IntegerField(source="job.id")

    class Meta:
        model = InferenceResult
        fields = ["job_id", "top_label", "output", "created_at"]
