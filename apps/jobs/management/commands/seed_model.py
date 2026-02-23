"""
seed_model.py
역할: 서버 최초 실행 시 ModelVersion 레코드를 DB에 등록하는 management command.
      이미 존재하면 건너뜀 (멱등성 보장 — 여러 번 실행해도 중복 생성 없음).
      Spring의 CommandLineRunner.run() 초기 데이터 세팅과 동일.

실행: python manage.py seed_model
"""

from django.core.management.base import BaseCommand
from apps.jobs.models import ModelVersion


class Command(BaseCommand):
    help = "Seed the initial ModelVersion record if it does not exist."

    def handle(self, *args, **kwargs):
        model_name = "densenet121-res224-all"

        # get_or_create: 없으면 INSERT, 있으면 SELECT — 중복 방지
        # (Spring의 findByNameOrElseSave()와 동일)
        obj, created = ModelVersion.objects.get_or_create(
            name=model_name,
            defaults={
                # 실제 가중치는 torchxrayvision이 HuggingFace에서 런타임에 로드
                # weights_path는 참조용 식별자로만 사용
                "weights_path": f"huggingface:{model_name}",
            },
        )

        if created:
            self.stdout.write(
                self.style.SUCCESS(f"[seed_model] Created ModelVersion: {model_name}")
            )
        else:
            self.stdout.write(
                f"[seed_model] ModelVersion already exists: {model_name} (skipped)"
            )
