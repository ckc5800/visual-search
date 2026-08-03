"""두 모델의 랭킹을 Reciprocal Rank Fusion(RRF)으로 융합해 평가한다.

동기: ko-clip과 en-clip-llm은 실패 문항이 서로 다르다(ko-clip은 "쿠르타",
LLM 번역은 "여성 원피스"). 랭킹 융합이 서로의 구멍을 메우는지 측정한다.

RRF(d) = Σ_m 1/(K + rank_m(d)), K=60 (표준값). 두 인덱스는 행 순서가 같아야
하며(같은 데이터셋 전체 인제스트) 시작 시 id 컬럼으로 검증한다.

사용:
    python src/ensemble.py --models ko-clip en-clip-llm [--hybrid] [--depth 100]
"""
import argparse
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from config import MODELS, ROOT
from evaluate import BUCKETS, bucket_of, build_gold_mask, load_queries, rank_metrics

RRF_K = 60


def rrf_fuse(rank_lists: list[np.ndarray], n_items: int, depth: int) -> np.ndarray:
    """rank_lists: 모델별 상위 depth 문서 인덱스 배열. RRF 점수 배열(n_items) 반환."""
    scores = np.zeros(n_items)
    for ranks in rank_lists:
        scores[ranks[:depth]] += 1.0 / (RRF_K + np.arange(len(ranks[:depth])) + 1)
    return scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs=2, default=["ko-clip", "en-clip-llm"],
                    choices=sorted(MODELS))
    ap.add_argument("--k", type=int, nargs="+", default=[1, 3, 5, 10])
    ap.add_argument("--depth", type=int, default=100, help="모델별 융합 후보 깊이")
    ap.add_argument("--hybrid", action="store_true")
    args = ap.parse_args()

    from hybrid import apply_filter_to_scores, extract_filters
    from models import load_embedder
    from vectorstore import VectorStore

    stores = [VectorStore.load(m) for m in args.models]
    if not (stores[0].meta["id"].to_numpy() == stores[1].meta["id"].to_numpy()).all():
        raise SystemExit("두 인덱스의 행 정렬이 다름 — 전체 인제스트끼리만 융합 가능")
    meta = stores[0].meta
    queries = load_queries()

    embedders = [load_embedder(m) for m in args.models]
    q_embs = [e.embed_texts([q["q"] for q in queries]) for e in embedders]

    max_k = max(args.k)
    rows = []
    for i, q in enumerate(queries):
        gold = build_gold_mask(meta, q["filters"])
        if not gold.any():
            continue
        hf = extract_filters(q["q"]) if args.hybrid else {}
        hmask = build_gold_mask(meta, hf) if hf else None
        rank_lists = []
        for store, q_emb in zip(stores, q_embs):
            scores = store.emb @ q_emb[i]
            if hmask is not None:
                scores = apply_filter_to_scores(scores, hmask)
            order = np.argsort(-scores)[: args.depth]
            rank_lists.append(order[np.isfinite(scores[order])])
        fused = rrf_fuse(rank_lists, len(meta), args.depth)
        top = np.argsort(-fused)[:max_k]
        top = top[fused[top] > 0]
        m = rank_metrics(gold[top], args.k)
        rows.append({"q": q["q"], "gold_n": int(gold.sum()),
                     "bucket": bucket_of(int(gold.sum())), **m})

    df = pd.DataFrame(rows)
    metric_cols = [f"hit@{k}" for k in args.k] + [f"precision@{max_k}", "rr"]
    summary = {c if c != "rr" else "MRR": float(df[c].mean()) for c in metric_cols}

    name = "+".join(args.models) + ("_hybrid" if args.hybrid else "")
    print(f"\n[ensemble] {name} (RRF K={RRF_K}, depth={args.depth}), 질의 {len(df)}건")
    for k, v in summary.items():
        print(f"  {k}: {v:.3f}")
    for _, _, bname in BUCKETS:
        sub = df[df["bucket"] == bname]
        if len(sub):
            print(f"  {bname} ({len(sub)}문항): hit@1 {sub['hit@1'].mean():.2f}  "
                  f"hit@{max_k} {sub[f'hit@{max_k}'].mean():.2f}  "
                  f"precision@{max_k} {sub[f'precision@{max_k}'].mean():.2f}")
    misses = df[~df[f"hit@{max_k}"]]
    if len(misses):
        print(f"  [miss] {misses['q'].tolist()}")

    out = ROOT / "eval" / f"results_ensemble_{name}.json"
    out.write_text(json.dumps({
        "run": {"timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "models": args.models, "rrf_k": RRF_K, "depth": args.depth,
                "hybrid": args.hybrid, "n_items": int(len(meta))},
        "summary": summary, "per_query": rows,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ensemble] 저장: {out}")


if __name__ == "__main__":
    main()
