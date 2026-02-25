"""
validate_model.py
역할: 실제 흉부 X-ray 샘플 이미지로 모델 출력이 임상적으로 의미있는지 검증.
      serving 인프라와 동일한 preprocess() → predict() 파이프라인 사용.
      실행: python scripts/validate_model.py
"""

import sys
import os
import io
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Django 없이 모델 로더만 사용 (독립 실행)
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")

# torchxrayvision 테스트 슈트에서 사용하는 NIH ChestX-ray14 샘플 이미지
SAMPLE_URL = (
    "https://raw.githubusercontent.com/mlmed/torchxrayvision"
    "/master/tests/00000001_000.png"
)


def download_sample(url: str) -> bytes:
    """URL에서 이미지 bytes 다운로드."""
    print(f"  Downloading: {url}")
    with urllib.request.urlopen(url, timeout=15) as resp:
        return resp.read()


def main():
    print("=" * 60)
    print("  Model Output Validation — Real Chest X-ray Sample")
    print("=" * 60)

    # ── 1. 이미지 준비 ────────────────────────────────────────
    # 커맨드라인 인자로 로컬 파일 경로를 받을 수 있음
    # 없으면 torchxrayvision GitHub 테스트 이미지 다운로드
    print("\n[1] Loading sample X-ray image...")
    if len(sys.argv) > 1:
        path = sys.argv[1]
        with open(path, "rb") as f:
            image_bytes = f.read()
        print(f"  Local file: {path} ({len(image_bytes):,} bytes)")
    else:
        try:
            image_bytes = download_sample(SAMPLE_URL)
            print(f"  Downloaded: {len(image_bytes):,} bytes")
        except Exception as e:
            print(f"  Download failed: {e}")
            print("  Usage: python scripts/validate_model.py [path/to/xray.png]")
            return

    # ── 2. 모델 로드 ──────────────────────────────────────────
    print("\n[2] Loading model...")
    from workers.model_loader import get_loader
    loader = get_loader()
    loader.load()
    print("  OK")

    # ── 3. 전처리 ─────────────────────────────────────────────
    # worker.py의 process_batch()와 동일한 파이프라인
    print("\n[3] Preprocessing (PNG [0,255] → tensor [-1024,1024])...")
    tensor = loader.preprocess(image_bytes)
    print(f"  tensor shape : {tuple(tensor.shape)}")
    print(f"  tensor range : [{tensor.min():.1f}, {tensor.max():.1f}]")

    # ── 4. 추론 ───────────────────────────────────────────────
    print("\n[4] Running inference (densenet121-res224-all)...")
    scores = loader.predict(tensor)
    print("  OK")

    # ── 5. 결과 출력 ──────────────────────────────────────────
    print("\n=== Pathology Scores (18 pathologies) ===\n")
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    for label, score in sorted_scores:
        bar = "█" * int(score * 40)       # 점수 시각화 (최대 40칸)
        print(f"  {label:<30} {score:.4f}  {bar}")

    top_label = sorted_scores[0][0]
    top_score = sorted_scores[0][1]

    print(f"\n  top_label : {top_label} ({top_score:.4f})")
    print(
        "\n  Note: Each score is an independent probability (multi-label model)."
        "\n  Non-zero scores across multiple pathologies are expected."
    )


if __name__ == "__main__":
    main()
