"""HF Space(YoonSeon/visual-search) 생성·배포.

사전 조건: `hf auth login`으로 write 토큰 로그인 (토큰은 huggingface_hub가 보관).
실행: .venv\\Scripts\\python.exe space\\deploy.py
"""
from pathlib import Path

from huggingface_hub import HfApi

REPO_ID = "YoonSeon/visual-search"
ROOT = Path(__file__).resolve().parent.parent

api = HfApi()
user = api.whoami()["name"]
print(f"로그인: {user}")

api.create_repo(REPO_ID, repo_type="space", space_sdk="docker", exist_ok=True)
print(f"Space 준비됨: https://huggingface.co/spaces/{REPO_ID}")

up_file = lambda src, dst: api.upload_file(
    path_or_fileobj=str(ROOT / src), path_in_repo=dst,
    repo_id=REPO_ID, repo_type="space")

up_file("space/README_space.md", "README.md")
up_file("space/Dockerfile", "Dockerfile")
up_file("space/requirements.txt", "requirements.txt")
print("메타 파일 업로드 완료")

for folder in ("src", "static"):
    api.upload_folder(folder_path=str(ROOT / folder), path_in_repo=folder,
                      repo_id=REPO_ID, repo_type="space",
                      ignore_patterns=["__pycache__/*", "*.pyc"])
print("코드 업로드 완료")

up_file("data/index/emb_ko-clip.npy", "data/index/emb_ko-clip.npy")
up_file("data/index/meta_ko-clip.parquet", "data/index/meta_ko-clip.parquet")
print("인덱스 업로드 완료 (~90MB)")

print(f"\n빌드 시작됨 — https://huggingface.co/spaces/{REPO_ID}")
