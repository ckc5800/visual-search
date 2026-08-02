"""한국어 문장형 질의로 상품 검색.

사용:
    python src/search.py "체크무늬 네이비 남성 셔츠"
    python src/search.py --model jina-v2 --top-k 10 "여름용 캐주얼 원피스"
"""
import argparse

from config import DEFAULT_MODEL, MODELS
from models import load_embedder
from vectorstore import VectorStore


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("--model", default=DEFAULT_MODEL, choices=sorted(MODELS))
    ap.add_argument("--top-k", type=int, default=5)
    args = ap.parse_args()

    store = VectorStore.load(args.model)
    embedder = load_embedder(args.model)
    q = embedder.embed_texts([args.query])[0]
    result = store.search(q, top_k=args.top_k)

    print(f'\n질의: "{args.query}" (model={args.model}, {len(store.meta)}개 상품 중 top-{args.top_k})\n')
    cols = ["score", "id", "articleType", "baseColour", "gender", "usage", "productDisplayName"]
    print(result[cols].to_string(index=False))


if __name__ == "__main__":
    main()
