"""검색 데모 API + 웹 UI.

실행:
    python -m uvicorn --app-dir src api:app --port 8000
    # 모델 선택: VS_MODEL=jina-v2 (기본 ko-clip)

- GET /               데모 페이지 (질의 입력 → 상품 이미지 그리드)
- GET /api/search     ?q=한국어질의&k=24 — 텍스트 검색
- GET /api/similar    ?pos=행번호&k=24 — 비슷한 상품 (인덱스의 저장 임베딩 재사용,
                      모델 추론 없음 — 이미지 클릭으로 진입)
- GET /image/{pos}    상품 이미지 (HF 데이터셋 캐시에서 직접 서빙, 썸네일 파일 불필요)
"""
import io
import os
import time
from contextlib import asynccontextmanager
from functools import lru_cache

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response

from config import DATASET_ID, DEFAULT_MODEL, ROOT

STATE = {}
RESULT_COLS = ["id", "articleType", "subCategory", "baseColour", "gender", "usage",
               "productDisplayName"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    from datasets import load_dataset

    from models import load_embedder
    from vectorstore import VectorStore

    model_key = os.environ.get("VS_MODEL", DEFAULT_MODEL)
    store = VectorStore.load(model_key)
    ds = load_dataset(DATASET_ID, split="train")
    if len(ds) != len(store.meta) or ds[0]["id"] != int(store.meta["id"].iloc[0]):
        raise RuntimeError("데이터셋과 인덱스의 행 정렬이 다름 — 전체 인제스트로 재구축 필요")
    STATE.update(model_key=model_key, store=store, ds=ds,
                 embedder=load_embedder(model_key))
    yield
    STATE.clear()


app = FastAPI(lifespan=lifespan)


def _results(indices, scores) -> list[dict]:
    meta = STATE["store"].meta
    out = []
    for pos in indices:
        row = meta.iloc[int(pos)]
        item = {}
        for c in RESULT_COLS:
            v = row[c]
            if v != v:  # NaN → null
                v = None
            elif hasattr(v, "item"):  # numpy 스칼라 → 파이썬 기본 타입
                v = v.item()
            item[c] = v
        item.update(pos=int(pos), score=round(float(scores[int(pos)]), 4))
        out.append(item)
    return out


@lru_cache(maxsize=512)
def _embed_query(q: str):
    """질의 임베딩 캐시 — 반복 질의(예시 칩, 정교화 재검색)의 TTFB를 줄인다.
    모델은 프로세스당 하나(VS_MODEL)라 질의 문자열만 키로 충분."""
    return STATE["embedder"].embed_texts([q])[0]


def _rank(q: str, filters: dict, k: int):
    """임베딩→랭킹 공통 경로. 단계별 소요 시간(ms)을 함께 반환한다."""
    import numpy as np

    from evaluate import build_gold_mask
    from hybrid import apply_filter_to_scores

    store = STATE["store"]
    t0 = time.perf_counter()
    emb = _embed_query(q)
    t1 = time.perf_counter()
    scores = store.emb @ emb
    if filters:
        scores = apply_filter_to_scores(scores, build_gold_mask(store.meta, filters))
    top = np.argsort(-scores)[:k]
    top = top[np.isfinite(scores[top])]  # 필터 통과분이 k 미만이면 그만큼만
    t2 = time.perf_counter()
    timing = {"embed_ms": round((t1 - t0) * 1000, 1), "rank_ms": round((t2 - t1) * 1000, 1)}
    return top, scores, timing


@app.get("/api/search")
def search(q: str, k: int = 24, hybrid: bool = False):
    from hybrid import extract_filters

    filters = extract_filters(q) if hybrid else {}
    top, scores, timing = _rank(q, filters, k)
    return {"model": STATE["model_key"], "query": q, "hybrid": hybrid,
            "filters": filters, "timing": timing, "items": _results(top, scores)}


@app.get("/api/refine")
def refine(q: str, followup: str, filters: str = "{}", k: int = 24):
    import json

    from refine import merge_refinement

    new_q, merged = merge_refinement(q, json.loads(filters), followup)
    top, scores, timing = _rank(new_q, merged, k)
    return {"model": STATE["model_key"], "query": new_q, "filters": merged,
            "timing": timing, "items": _results(top, scores)}


@app.get("/api/similar")
def similar(pos: int, k: int = 24):
    import numpy as np

    store = STATE["store"]
    if not 0 <= pos < len(store.meta):
        raise HTTPException(404)
    scores = store.emb @ store.emb[pos]
    top = np.argsort(-scores)[1 : k + 1]  # 자기 자신 제외
    return {"model": STATE["model_key"], "anchor": _results([pos], scores)[0],
            "items": _results(top, scores)}


@app.get("/image/{pos}")
def image(pos: int):
    ds = STATE["ds"]
    if not 0 <= pos < len(ds):
        raise HTTPException(404)
    buf = io.BytesIO()
    ds[pos]["image"].convert("RGB").save(buf, format="JPEG", quality=85)
    return Response(buf.getvalue(), media_type="image/jpeg",
                    headers={"Cache-Control": "max-age=86400"})


@app.get("/")
def index():
    return FileResponse(ROOT / "static" / "index.html")
