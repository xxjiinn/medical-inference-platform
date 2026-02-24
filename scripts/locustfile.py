"""
locustfile.py
역할: 병원 X-ray 추론 시스템 부하 테스트.
      두 가지 시나리오를 동시에 측정:
        - cache_miss: 매번 다른 이미지 → 워커가 실제로 추론
        - cache_hit:  동일 이미지 반복 → SHA256 캐시로 즉시 반환

      end-to-end inference latency (제출 → COMPLETED)를 Locust 커스텀 이벤트로 직접 보고.
      Locust 내장 response_time은 단일 HTTP 요청 기준이라 추론 latency와 다름.

실행: locust -f scripts/locustfile.py --host http://<EC2_IP>:8000
"""

import io
import time
import numpy as np
from PIL import Image
from locust import HttpUser, task, between


# SHA256 캐시 히트 시나리오용 고정 이미지 (항상 같은 픽셀 → 같은 SHA256)
# 첫 번째 요청에서 COMPLETED 처리된 뒤, 이후 요청은 캐시 히트로 즉시 반환됨
_FIXED_IMAGE_CACHE: bytes | None = None


def make_xray_image(fixed: bool = False) -> bytes:
    """
    224×224 흑백 더미 이미지 생성.
    fixed=True: 항상 같은 이미지 (캐시 히트 시나리오)
    fixed=False: 랜덤 노이즈 (매 요청마다 새 이미지, SHA256 중복 방지)
    """
    global _FIXED_IMAGE_CACHE

    if fixed:
        if _FIXED_IMAGE_CACHE is None:
            # 첫 호출 시 고정 이미지 생성 후 캐싱 (이후 재사용)
            pixel = np.full((224, 224), 128, dtype=np.uint8)
            img = Image.fromarray(pixel, mode="L")
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            _FIXED_IMAGE_CACHE = buf.getvalue()
        return _FIXED_IMAGE_CACHE

    noise = np.random.randint(0, 256, (224, 224), dtype=np.uint8)
    img = Image.fromarray(noise, mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class HospitalUser(HttpUser):
    """
    병원 PACS 시스템의 요청 패턴 시뮬레이션.
    X-ray 업로드 → 결과 폴링 흐름.
    wait_time: 요청 사이 1~3초 대기 (실제 PACS 인터벌 모사)
    """
    wait_time = between(1, 3)

    def on_start(self):
        """
        테스트 시작 전 고정 이미지를 미리 처리해 캐시를 워밍업.
        이 작업 없이 부하 테스트를 시작하면 cache_hit 요청이 COMPLETED가
        아닌 QUEUED 상태의 job을 기다리다 타임아웃됨.
        첫 번째 유저만 실제로 제출하고 나머지는 캐시를 자연스럽게 재사용.
        """
        image_bytes = make_xray_image(fixed=True)
        resp = self.client.post(
            "/v1/jobs",
            files={"image": ("xray.png", image_bytes, "image/png")},
            name="/v1/jobs (warmup)",
        )
        if resp.status_code not in (200, 201):
            return
        job_id = resp.json().get("id")

        # 워밍업 job이 COMPLETED될 때까지 대기 (최대 30초)
        deadline = time.time() + 30
        while time.time() < deadline:
            res = self.client.get(f"/v1/jobs/{job_id}", name="/v1/jobs/[id] (warmup)")
            if res.json().get("status") == "COMPLETED":
                break
            time.sleep(0.2)

    def _submit_and_wait(self, image_bytes: bytes, label: str) -> None:
        """
        이미지 제출 후 COMPLETED/FAILED 될 때까지 폴링.
        end-to-end inference latency를 직접 계산해 Locust 이벤트로 보고.

        Locust 내장 response_time과 별도로 측정하는 이유:
          내장값 = 단일 HTTP 요청 시간 (수십 ms)
          여기서 측정 = 제출부터 추론 완료까지 총 대기 시간 (수백 ms ~ 수 초)
        """
        start = time.perf_counter()

        # 1. 이미지 업로드 → job_id 수신
        with self.client.post(
            "/v1/jobs",
            files={"image": ("xray.png", image_bytes, "image/png")},
            catch_response=True,
            name="/v1/jobs",
        ) as resp:
            if resp.status_code not in (200, 201):
                resp.failure(f"job 생성 실패: {resp.status_code}")
                return
            job_id = resp.json().get("id")
            resp.success()

        # 2. COMPLETED/FAILED 될 때까지 폴링 (최대 15초, 100ms 간격)
        #    100ms로 좁혀야 추론 latency를 정확하게 측정할 수 있음 (0.5s는 오차 큼)
        deadline = time.time() + 15
        final_status = None
        while time.time() < deadline:
            res = self.client.get(f"/v1/jobs/{job_id}", name="/v1/jobs/[id]")
            final_status = res.json().get("status")
            if final_status in ("COMPLETED", "FAILED"):
                break
            time.sleep(0.1)

        elapsed_ms = (time.perf_counter() - start) * 1000

        # 3. end-to-end inference latency를 Locust 커스텀 이벤트로 보고
        #    Locust 보고서에 "inference/cache_miss", "inference/cache_hit"으로 분리 표시
        self.environment.events.request.fire(
            request_type="inference",
            name=f"e2e/{label}",
            response_time=elapsed_ms,
            response_length=0,
            exception=None if final_status == "COMPLETED" else Exception(f"status={final_status}"),
        )

    @task(7)
    def new_image(self):
        """
        랜덤 이미지 제출 — SHA256 캐시 미스.
        워커가 실제로 추론해야 하는 일반 경로.
        전체 요청의 70%를 차지 (7 : 3 비율).
        """
        self._submit_and_wait(make_xray_image(fixed=False), label="cache_miss")

    @task(3)
    def cached_image(self):
        """
        동일 이미지 반복 제출 — SHA256 캐시 히트.
        POST /v1/jobs에서 캐시 히트 시 즉시 기존 job_id 반환 → 이미 COMPLETED이므로 대기 없음.
        전체 요청의 30%.
        latency가 cache_miss 대비 10~20배 짧으면 캐시 효과 입증.
        """
        self._submit_and_wait(make_xray_image(fixed=True), label="cache_hit")
