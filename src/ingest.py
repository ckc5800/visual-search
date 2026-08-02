"""상품 이미지 임베딩 인덱스 구축.

사용:
    python src/ingest.py                     # 전체 44K, 기본 모델(ko-clip)
    python src/ingest.py --limit 200         # 스모크: 앞 200개만 (스트리밍, 전체 다운로드 없음)
    python src/ingest.py --model jina-v2

산출물: data/index/emb_<model>.npy, meta_<model>.parquet (행 순서로 정렬 일치)
"""
import argparse

import numpy as np
import pandas as pd
from tqdm import tqdm

from config import DATASET_ID, DEFAULT_MODEL, INDEX_DIR, MODELS, index_paths
from models import load_embedder

META_COLS = [
    "id", "gender", "masterCategory", "subCategory", "articleType",
    "baseColour", "season", "year", "usage", "productDisplayName",
]


def load_items(limit: int | None):
    from datasets import load_dataset

    if limit:
        ds = load_dataset(DATASET_ID, split="train", streaming=True)
        rows = [r for _, r in zip(range(limit), ds)]
    else:
        rows = load_dataset(DATASET_ID, split="train")
    images, meta = [], []
    for r in rows:
        images.append(r["image"].convert("RGB"))
        meta.append({c: r[c] for c in META_COLS})
    return images, pd.DataFrame(meta)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL, choices=sorted(MODELS))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    print(f"[ingest] dataset={DATASET_ID} limit={args.limit or 'all'} model={args.model}")
    images, meta = load_items(args.limit)
    print(f"[ingest] {len(images)}개 이미지 로드 완료, 임베딩 시작")

    embedder = load_embedder(args.model)
    print(f"[ingest] device={embedder.device}")

    # tqdm을 위해 배치 단위로 직접 순회
    bs = args.batch_size
    chunks = [
        embedder.embed_images(images[i : i + bs], batch_size=bs)
        for i in tqdm(range(0, len(images), bs), desc="embed")
    ]
    emb = np.concatenate(chunks).astype(np.float32)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    emb_path, meta_path = index_paths(args.model)
    np.save(emb_path, emb)
    meta.to_parquet(meta_path, index=False)
    print(f"[ingest] 저장: {emb_path} {emb.shape}, {meta_path} {len(meta)}행")


if __name__ == "__main__":
    main()
