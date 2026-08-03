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


def iter_rows(limit: int | None):
    """(rows, total) — limit이 있으면 스트리밍(전체 다운로드 없음), 없으면 캐시 전체."""
    from datasets import load_dataset

    if limit:
        ds = load_dataset(DATASET_ID, split="train", streaming=True)
        return (r for _, r in zip(range(limit), ds)), limit
    ds = load_dataset(DATASET_ID, split="train")
    return iter(ds), len(ds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL, choices=sorted(MODELS))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    print(f"[ingest] dataset={DATASET_ID} limit={args.limit or 'all'} model={args.model}")
    embedder = load_embedder(args.model)
    print(f"[ingest] device={embedder.device}")

    # 이미지를 전부 메모리에 올리지 않고 배치 단위로 디코드→임베딩→해제
    # (44K 전체를 PIL로 들고 있으면 1GB 이상 점유)
    rows, total = iter_rows(args.limit)
    bs = args.batch_size
    chunks, meta, batch = [], [], []
    with tqdm(total=total, desc="embed") as bar:
        for r in rows:
            batch.append(r["image"].convert("RGB"))
            meta.append({c: r[c] for c in META_COLS})
            if len(batch) == bs:
                chunks.append(embedder.embed_images(batch, batch_size=bs))
                bar.update(len(batch))
                batch = []
        if batch:
            chunks.append(embedder.embed_images(batch, batch_size=bs))
            bar.update(len(batch))
    emb = np.concatenate(chunks).astype(np.float32)
    meta = pd.DataFrame(meta)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    emb_path, meta_path = index_paths(args.model)
    np.save(emb_path, emb)
    meta.to_parquet(meta_path, index=False)
    print(f"[ingest] 저장: {emb_path} {emb.shape}, {meta_path} {len(meta)}행")


if __name__ == "__main__":
    main()
