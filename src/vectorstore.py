"""임베딩 인덱스 로드와 top-k 검색.

44K 규모라 정확 탐색(numpy 내적)으로 충분하다 — 임베딩이 L2 정규화돼 있어
내적 = 코사인 유사도. ANN(FAISS 등)은 규모가 커져 필요해질 때 도입한다.
"""
import numpy as np
import pandas as pd

from config import index_paths


class VectorStore:
    def __init__(self, emb: np.ndarray, meta: pd.DataFrame):
        if len(emb) != len(meta):
            raise ValueError(f"임베딩 {len(emb)}행 != 메타 {len(meta)}행 — 인덱스 재구축 필요")
        self.emb = emb
        self.meta = meta

    @classmethod
    def load(cls, model_key: str) -> "VectorStore":
        emb_path, meta_path = index_paths(model_key)
        if not emb_path.exists():
            raise FileNotFoundError(
                f"{emb_path} 없음 — 먼저 `python src/ingest.py --model {model_key}` 실행"
            )
        return cls(np.load(emb_path), pd.read_parquet(meta_path))

    def search(self, query_emb: np.ndarray, top_k: int = 5) -> pd.DataFrame:
        """query_emb: L2 정규화된 (D,) 또는 (1, D). score 내림차순 DataFrame 반환."""
        q = query_emb.reshape(-1)
        scores = self.emb @ q
        idx = np.argsort(-scores)[:top_k]
        out = self.meta.iloc[idx].copy()
        out.insert(0, "score", scores[idx])
        return out.reset_index(drop=True)
