"""하이브리드 검색: 한국어 질의에서 구조화 조건(색·성별·용도)을 룰 기반으로 추출해
메타 필터로 걸고, 임베딩은 필터 통과분 안에서 순위를 매긴다.

평가에서 확인된 문제("좁은 조건일수록 임베딩 단독 precision 급락)에 대한 대응.
카테고리(셔츠/시계 등)는 임베딩이 이미 잘 맞히므로 추출하지 않는다 — 필터는
임베딩이 자주 틀리는 속성(색·성별·용도)만 담당한다.

LLM 추출로 바꿀 수 있는 구조지만, 속성 어휘가 닫혀 있는 도메인이라 사전 매핑으로
시작한다 (오추출 디버깅이 쉽고 지연이 0에 가깝다).
"""
import re

import numpy as np

# "네이비 말고", "검정 빼고", "정장 제외" — 부정된 단어는 추출에서 뺀다
_NEG = re.compile(r"(\S+?)\s*(?:말고|빼고|제외)")

# 긴 패턴을 먼저 검사한다 ("연분홍"이 "분홍"보다 먼저 매칭되도록 정렬은 아래에서)
COLOUR_MAP = {
    "네이비": "Navy Blue", "남색": "Navy Blue",
    "검정": "Black", "검은": "Black", "블랙": "Black",
    "흰색": "White", "하얀": "White", "화이트": "White",
    "파란": "Blue", "파랑": "Blue", "블루": "Blue", "청색": "Blue",
    "빨간": "Red", "빨강": "Red", "레드": "Red",
    "초록": "Green", "녹색": "Green", "그린": "Green",
    "회색": "Grey", "그레이": "Grey",
    "분홍": "Pink", "핑크": "Pink",
    "노란": "Yellow", "노랑": "Yellow", "옐로": "Yellow",
    "보라": "Purple", "퍼플": "Purple",
    "갈색": "Brown", "브라운": "Brown",
    "실버": "Silver", "은색": "Silver",
    "골드": "Gold", "금색": "Gold",
    "베이지": "Beige", "주황": "Orange", "오렌지": "Orange",
}
GENDER_MAP = {
    "남아": "Boys", "여아": "Girls",
    "남성": "Men", "남자": "Men",
    "여성": "Women", "여자": "Women",
    "아이들": ("Boys", "Girls"), "키즈": ("Boys", "Girls"),
}
USAGE_MAP = {
    "캐주얼": "Casual",
    "정장": "Formal", "출근": "Formal", "포멀": "Formal",
    "운동": "Sports", "스포츠": "Sports",
    "파티": "Party",
    "에스닉": "Ethnic", "전통": "Ethnic",
}


def _match(query: str, mapping: dict) -> list[str]:
    hits = set()
    for k, v in mapping.items():
        if k in query:
            hits.update(v if isinstance(v, tuple) else (v,))
    return sorted(hits)


FIELD_MAPS = {"baseColour": COLOUR_MAP, "gender": GENDER_MAP, "usage": USAGE_MAP}


def extract_filters(query: str) -> dict:
    """질의에서 메타 필터 조건 추출. "X 말고/빼고/제외"의 X는 제외. 없으면 빈 dict.
    같은 필드에 여러 값이 잡히면 OR로 남긴다 (좁히기보다 안전한 쪽)."""
    neg_words = " ".join(_NEG.findall(query))
    out = {}
    for field, mapping in FIELD_MAPS.items():
        vals = sorted(set(_match(query, mapping)) - set(_match(neg_words, mapping)))
        if vals:
            out[field] = vals
    return out


def apply_filter_to_scores(scores: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """필터 통과분만 순위 대상. 통과분이 0이면 필터를 무시(폴백)한다."""
    if not mask.any():
        return scores
    return np.where(mask, scores, -np.inf)
