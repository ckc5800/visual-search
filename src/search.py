"""한국어 문장형 질의 또는 이미지 파일로 상품 검색.

사용:
    python src/search.py "체크무늬 네이비 남성 셔츠"
    python src/search.py --model jina-v2 --top-k 10 "여름용 캐주얼 원피스"
    python src/search.py --image my_shirt.jpg          # 비슷한 상품 찾기
"""
import argparse

from config import DEFAULT_MODEL, MODELS
from models import load_embedder
from vectorstore import VectorStore


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query", nargs="?", default=None)
    ap.add_argument("--image", default=None, help="질의 대신 이미지 파일로 검색")
    ap.add_argument("--model", default=DEFAULT_MODEL, choices=sorted(MODELS))
    ap.add_argument("--top-k", type=int, default=5)
    args = ap.parse_args()
    if bool(args.query) == bool(args.image):
        ap.error("질의 텍스트 또는 --image 중 하나만 지정")

    store = VectorStore.load(args.model)
    embedder = load_embedder(args.model)
    if args.image:
        from PIL import Image

        q = embedder.embed_images([Image.open(args.image).convert("RGB")])[0]
        label = f"이미지 {args.image}"
    else:
        q = embedder.embed_texts([args.query])[0]
        label = f'"{args.query}"'
    result = store.search(q, top_k=args.top_k)

    print(f"\n질의: {label} (model={args.model}, {len(store.meta)}개 상품 중 top-{args.top_k})\n")
    cols = ["score", "id", "articleType", "baseColour", "gender", "usage", "productDisplayName"]
    print(result[cols].to_string(index=False))


if __name__ == "__main__":
    main()
