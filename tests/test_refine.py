import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from hybrid import extract_filters  # noqa: E402
from refine import merge_refinement  # noqa: E402


def test_negation_in_extract():
    assert extract_filters("네이비 말고 검정색으로") == {"baseColour": ["Black"]}
    assert extract_filters("정장 빼고 캐주얼") == {"usage": ["Casual"]}


def test_colour_replace_keeps_category():
    q, f = merge_refinement(
        "네이비 남성 셔츠", {"baseColour": ["Navy Blue"], "gender": ["Men"]}, "검정색으로"
    )
    assert f == {"baseColour": ["Black"], "gender": ["Men"]}
    assert "네이비" not in q and "셔츠" in q and "검정" in q


def test_usage_added():
    q, f = merge_refinement("남성 셔츠", {"gender": ["Men"]}, "좀 더 캐주얼하게")
    assert f == {"gender": ["Men"], "usage": ["Casual"]}
    assert "셔츠" in q


def test_gender_replace():
    _, f = merge_refinement(
        "남성 운동화", {"gender": ["Men"]}, "여성용으로 보여줘"
    )
    assert f["gender"] == ["Women"]


def test_negation_followup():
    q, f = merge_refinement(
        "네이비 남성 셔츠", {"baseColour": ["Navy Blue"], "gender": ["Men"]},
        "네이비 말고 초록색"
    )
    assert f["baseColour"] == ["Green"]
    assert "말고" not in q
