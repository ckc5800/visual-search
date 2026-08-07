import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from hybrid import apply_filter_to_scores, extract_filters  # noqa: E402


def test_extract_colour_gender_usage():
    f = extract_filters("실버 여성 손목시계")
    assert f == {"baseColour": ["Silver"], "gender": ["Women"]}
    f = extract_filters("출근용 검정 남성 구두")
    assert f == {"baseColour": ["Black"], "gender": ["Men"], "usage": ["Formal"]}


def test_extract_none():
    assert extract_filters("선글라스") == {}


def test_extract_multi_value():
    f = extract_filters("아이들 티셔츠")
    assert f["gender"] == ["Boys", "Girls"]
    # 남아 ≠ 남자/남성 오탐 없음
    assert extract_filters("남아용 신발")["gender"] == ["Boys"]


def test_no_false_positive_in_compound_words():
    # "파란만장", "골드미스", "그린란드"는 색상 키워드를 부분 문자열로 포함하지만
    # 무관한 복합어 — 오추출되면 안 된다.
    assert extract_filters("파란만장한 인생 티셔츠") == {}
    assert extract_filters("골드미스룩 원피스") == {}
    assert extract_filters("그린란드 여행 자켓") == {}


def test_apply_filter_fallback_on_empty():
    scores = np.array([0.9, 0.5, 0.1], dtype=np.float32)
    empty = np.zeros(3, dtype=bool)
    assert np.array_equal(apply_filter_to_scores(scores, empty), scores)
    mask = np.array([False, True, False])
    out = apply_filter_to_scores(scores, mask)
    assert out[1] == np.float32(0.5) and np.isneginf(out[0]) and np.isneginf(out[2])
