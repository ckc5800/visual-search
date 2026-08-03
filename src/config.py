"""프로젝트 공통 설정. 경로·모델 레지스트리는 여기서만 정의한다."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
INDEX_DIR = DATA_DIR / "index"

DATASET_ID = "ashraq/fashion-product-images-small"

# 모델 레지스트리. key가 CLI --model 값이 된다.
# ko-clip: 한국어 특화 CLIP, MIT, 151M — 기본 베이스라인.
# jina-v2: 89개 언어 멀티링구얼, 865M, CC-BY-NC(비상업) — 비교군.
#          trust_remote_code가 필요해서 명시적으로 선택했을 때만 로드한다.
#          이미지 타워가 512px EVA02라 CPU에서 이미지당 ~7초 — 서브셋(--stride)으로만 실용적.
# m-clip: 멀티링구얼 텍스트 타워(mUSE 증류) + 영어 CLIP ViT-B/32 이미지 타워 조합,
#         apache-2.0 — 이미지 비용은 ko-clip과 동일.
MODELS = {
    "ko-clip": {
        "hf_id": "Bingsu/clip-vit-base-patch32-ko",
        "loader": "clip",
    },
    "jina-v2": {
        "hf_id": "jinaai/jina-clip-v2",
        "loader": "jina",
    },
    "m-clip": {
        "hf_id": "sentence-transformers/clip-ViT-B-32-multilingual-v1",
        "image_hf_id": "sentence-transformers/clip-ViT-B-32",
        "loader": "mclip",
    },
}
DEFAULT_MODEL = "ko-clip"

# 인덱스 파일 이름 규칙: 모델별로 분리 저장해 A/B를 가능하게 한다.
def index_paths(model_key: str):
    return (
        INDEX_DIR / f"emb_{model_key}.npy",
        INDEX_DIR / f"meta_{model_key}.parquet",
    )
