import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from vectorstore import VectorStore  # noqa: E402


def make_store():
    # 4개 축 방향 단위벡터 — 정답이 자명한 인덱스
    emb = np.eye(4, dtype=np.float32)
    meta = pd.DataFrame(
        {
            "id": [10, 11, 12, 13],
            "articleType": ["Shirts", "Jeans", "Watches", "Dresses"],
            "productDisplayName": ["s", "j", "w", "d"],
        }
    )
    return VectorStore(emb, meta)


def test_search_returns_exact_match_first():
    store = make_store()
    q = np.array([0, 0, 1, 0], dtype=np.float32)
    result = store.search(q, top_k=2)
    assert result.iloc[0]["id"] == 12
    assert result.iloc[0]["score"] == pytest.approx(1.0)
    assert len(result) == 2


def test_search_scores_descending():
    store = make_store()
    q = np.array([0.9, 0.1, 0, 0], dtype=np.float32)
    scores = store.search(q, top_k=4)["score"].tolist()
    assert scores == sorted(scores, reverse=True)


def test_length_mismatch_rejected():
    emb = np.eye(3, dtype=np.float32)
    meta = pd.DataFrame({"id": [1, 2]})
    with pytest.raises(ValueError):
        VectorStore(emb, meta)


def test_l2_normalize():
    from models import _l2_normalize

    x = np.array([[3.0, 4.0], [0.0, 0.0]], dtype=np.float32)
    out = _l2_normalize(x)
    assert np.linalg.norm(out[0]) == pytest.approx(1.0)
    assert not np.any(np.isnan(out))  # 영벡터도 NaN 없이 처리
