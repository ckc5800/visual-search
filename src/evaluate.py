"""한국어 질의 평가: 메타데이터 기반 gold 라벨로 hit@k·precision@k·MRR 측정.

gold = 필터(articleType·baseColour·gender·usage·season·subCategory AND 결합,
필드 값이 리스트면 OR)에 맞는 전체 상품 집합.

지표 설계:
- hit@k: top-k 안에 gold가 하나라도 있으면 1. 단, gold가 수천 개인 넓은 질의는
  hit이 거의 자동으로 나오므로 gold 크기 구간별로 분해해 함께 보고한다.
- precision@k: top-k 중 gold 비율 — 넓은 질의에서도 변별력이 있는 보조 지표.
- MRR: 첫 gold의 역순위 (top max_k 밖이면 0).

사용:
    python src/evaluate.py                      # 기본 모델, k=1,3,5,10
    python src/evaluate.py --model jina-v2 --strict
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from config import DATASET_ID, DEFAULT_MODEL, MODELS, ROOT

QUERIES_PATH = ROOT / "eval" / "queries_ko.jsonl"
FILTER_COLS = {"articleType", "subCategory", "baseColour", "gender", "usage", "season"}
# gold 크기 구간: 세부 조건(희소) / 중간 / 넓은 카테고리
BUCKETS = [(1, 50, "gold≤50"), (51, 500, "51~500"), (501, 10**9, "500+")]


def load_queries(path: Path = QUERIES_PATH) -> list[dict]:
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    return [json.loads(ln) for ln in lines if ln.strip()]


def build_gold_mask(meta: pd.DataFrame, filters: dict) -> np.ndarray:
    unknown = set(filters) - FILTER_COLS
    if unknown:
        raise ValueError(f"지원하지 않는 필터 필드: {unknown}")
    mask = np.ones(len(meta), dtype=bool)
    for col, val in filters.items():
        vals = val if isinstance(val, list) else [val]
        mask &= meta[col].isin(vals).to_numpy()
    return mask


def rank_metrics(ranked_gold: np.ndarray, ks: list[int]) -> dict:
    """ranked_gold: 점수 내림차순으로 정렬된 gold 여부 bool 배열 (상위 max(ks)개)."""
    max_k = max(ks)
    m = {f"hit@{k}": bool(ranked_gold[:k].any()) for k in ks}
    m[f"precision@{max_k}"] = float(ranked_gold[:max_k].mean())
    first = np.flatnonzero(ranked_gold)
    m["rr"] = float(1.0 / (first[0] + 1)) if len(first) else 0.0
    return m


def bucket_of(gold_n: int) -> str:
    for lo, hi, name in BUCKETS:
        if lo <= gold_n <= hi:
            return name
    raise ValueError(f"gold_n={gold_n}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL, choices=sorted(MODELS))
    ap.add_argument("--k", type=int, nargs="+", default=[1, 3, 5, 10])
    ap.add_argument("--strict", action="store_true", help="gold가 빈 질의가 있으면 실패")
    ap.add_argument("--subset-stride", type=int, default=1,
                    help="인덱스를 N행마다 1개로 슬라이스 — stride 인제스트된 모델과 공정 비교용")
    args = ap.parse_args()

    from models import load_embedder
    from vectorstore import VectorStore

    store = VectorStore.load(args.model)
    if args.subset_stride > 1:
        s = args.subset_stride
        store = VectorStore(store.emb[::s].copy(),
                            store.meta.iloc[::s].reset_index(drop=True))
    queries = load_queries()

    empty = [q["q"] for q in queries if not build_gold_mask(store.meta, q["filters"]).any()]
    if empty:
        print(f"[경고] gold 0개 질의 {len(empty)}건: {empty}")
        if args.strict:
            raise SystemExit(1)

    embedder = load_embedder(args.model)
    q_emb = embedder.embed_texts([q["q"] for q in queries])

    max_k = max(args.k)
    ids = store.meta["id"].to_numpy()
    rows = []
    for q, emb in zip(queries, q_emb):
        gold = build_gold_mask(store.meta, q["filters"])
        if not gold.any():
            continue
        scores = store.emb @ emb
        top = np.argsort(-scores)[:max_k]
        m = rank_metrics(gold[top], args.k)
        rows.append({
            "q": q["q"], "gold_n": int(gold.sum()), "bucket": bucket_of(int(gold.sum())),
            **m, "top_ids": ids[top].tolist(),
        })

    df = pd.DataFrame(rows)
    metric_cols = [f"hit@{k}" for k in args.k] + [f"precision@{max_k}", "rr"]
    summary = {c if c != "rr" else "MRR": float(df[c].mean()) for c in metric_cols}

    print(f"\n[evaluate] model={args.model}, 질의 {len(df)}건 (제외 {len(empty)}건), 상품 {len(store.meta)}개")
    for k, v in summary.items():
        print(f"  {k}: {v:.3f}")

    print("\n[gold 크기별 분해]")
    by_bucket = {}
    for _, _, name in BUCKETS:
        sub = df[df["bucket"] == name]
        if not len(sub):
            continue
        by_bucket[name] = {"n": len(sub), **{c if c != "rr" else "MRR": float(sub[c].mean()) for c in metric_cols}}
        b = by_bucket[name]
        print(f"  {name} ({b['n']}문항): hit@1 {b['hit@1']:.2f}  hit@{max_k} {b[f'hit@{max_k}']:.2f}  "
              f"precision@{max_k} {b[f'precision@{max_k}']:.2f}  MRR {b['MRR']:.2f}")

    misses = df[~df[f"hit@{max_k}"]]
    if len(misses):
        print(f"\n[miss] hit@{max_k} 실패 {len(misses)}건:")
        print(misses[["q", "gold_n"]].to_string(index=False))

    result = {
        "run": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "model_key": args.model,
            "model_hf_id": MODELS[args.model]["hf_id"],
            "dataset": DATASET_ID,
            "n_items": int(len(store.meta)),
            "emb_dim": int(store.emb.shape[1]),
            "subset_stride": args.subset_stride,
            "k": args.k,
        },
        "summary": summary,
        "by_bucket": by_bucket,
        "per_query": rows,
    }
    suffix = f"_stride{args.subset_stride}" if args.subset_stride > 1 else ""
    out = ROOT / "eval" / f"results_{args.model}{suffix}.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[evaluate] 저장: {out}")


if __name__ == "__main__":
    main()
