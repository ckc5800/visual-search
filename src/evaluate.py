"""한국어 질의 평가: 메타데이터 기반 gold 라벨로 hit@k·MRR 측정.

gold = 필터(articleType·baseColour·gender·usage·season·subCategory AND 결합,
필드 값이 리스트면 OR)에 맞는 전체 상품 집합. hit@k는 top-k 안에 gold가
하나라도 있으면 1 — 관련 상품이 수백 개인 검색 과제라 recall 대신 hit 기준.

사용:
    python src/evaluate.py                      # 기본 모델, k=1,3,5,10
    python src/evaluate.py --model jina-v2 --strict
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from config import DEFAULT_MODEL, MODELS, ROOT

QUERIES_PATH = ROOT / "eval" / "queries_ko.jsonl"
FILTER_COLS = {"articleType", "subCategory", "baseColour", "gender", "usage", "season"}


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
    hits = {f"hit@{k}": bool(ranked_gold[:k].any()) for k in ks}
    first = np.flatnonzero(ranked_gold)
    hits["rr"] = float(1.0 / (first[0] + 1)) if len(first) else 0.0
    return hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL, choices=sorted(MODELS))
    ap.add_argument("--k", type=int, nargs="+", default=[1, 3, 5, 10])
    ap.add_argument("--strict", action="store_true", help="gold가 빈 질의가 있으면 실패")
    args = ap.parse_args()

    from models import load_embedder
    from vectorstore import VectorStore

    store = VectorStore.load(args.model)
    queries = load_queries()

    empty = [q["q"] for q in queries if not build_gold_mask(store.meta, q["filters"]).any()]
    if empty:
        print(f"[경고] gold 0개 질의 {len(empty)}건: {empty}")
        if args.strict:
            raise SystemExit(1)

    embedder = load_embedder(args.model)
    q_emb = embedder.embed_texts([q["q"] for q in queries])

    max_k = max(args.k)
    rows = []
    for q, emb in zip(queries, q_emb):
        gold = build_gold_mask(store.meta, q["filters"])
        if not gold.any():
            continue
        scores = store.emb @ emb
        top = np.argsort(-scores)[:max_k]
        m = rank_metrics(gold[top], args.k)
        rows.append({"q": q["q"], "gold_n": int(gold.sum()), **m})

    df = pd.DataFrame(rows)
    summary = {f"hit@{k}": float(df[f"hit@{k}"].mean()) for k in args.k}
    summary["MRR"] = float(df["rr"].mean())

    print(f"\n[evaluate] model={args.model}, 질의 {len(df)}건 (제외 {len(empty)}건), 상품 {len(store.meta)}개")
    for k, v in summary.items():
        print(f"  {k}: {v:.3f}")
    misses = df[~df[f"hit@{max_k}"]]
    if len(misses):
        print(f"\n[miss] hit@{max_k} 실패 {len(misses)}건:")
        print(misses[["q", "gold_n"]].to_string(index=False))

    out = ROOT / "eval" / f"results_{args.model}.json"
    out.write_text(
        json.dumps({"model": args.model, "n_queries": len(df), "summary": summary,
                    "per_query": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n[evaluate] 저장: {out}")


if __name__ == "__main__":
    main()
