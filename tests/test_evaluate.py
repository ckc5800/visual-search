import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from evaluate import QUERIES_PATH, build_gold_mask, load_queries, rank_metrics  # noqa: E402


def meta():
    return pd.DataFrame(
        {
            "articleType": ["Shirts", "Shirts", "Jeans", "Watches"],
            "baseColour": ["Navy Blue", "White", "Blue", "Silver"],
            "gender": ["Men", "Men", "Women", "Women"],
            "usage": ["Casual", "Formal", "Casual", "Casual"],
            "season": ["Fall"] * 4,
            "subCategory": ["Topwear", "Topwear", "Bottomwear", "Watches"],
        }
    )


def test_gold_mask_and_combination():
    m = build_gold_mask(meta(), {"articleType": "Shirts", "gender": "Men"})
    assert m.tolist() == [True, True, False, False]
    # 리스트 값은 OR
    m = build_gold_mask(meta(), {"baseColour": ["Navy Blue", "Silver"]})
    assert m.tolist() == [True, False, False, True]


def test_gold_mask_rejects_unknown_field():
    with pytest.raises(ValueError):
        build_gold_mask(meta(), {"price": "cheap"})


def test_rank_metrics():
    ranked = np.array([False, True, False, True])
    m = rank_metrics(ranked, ks=[1, 3])
    assert m["hit@1"] is False and m["hit@3"] is True
    assert m["rr"] == pytest.approx(0.5)
    assert rank_metrics(np.array([False, False]), ks=[1])["rr"] == 0.0


def test_queries_file_valid():
    queries = load_queries(QUERIES_PATH)
    assert len(queries) == 50
    for q in queries:
        assert q["q"].strip()
        assert q["filters"]
        # 필드 이름 오타는 여기서 잡는다
        build_gold_mask(meta(), {k: v for k, v in q["filters"].items()})
