"""대화형 정교화: 이전 검색 상태(질의 텍스트 + 병합 필터)에 후속 발화를 병합한다.

규칙:
- 후속 발화에서 추출된 필드(색·성별·용도)는 이전 값을 교체한다.
  ("네이비 셔츠" → "검정색으로" = 색만 Black으로, 카테고리 유지)
- 임베딩용 질의 텍스트는 교체된 필드의 어휘를 이전 질의에서 지우고
  후속 발화(부정 구문 "X 말고"는 제거)를 이어 붙인다. "검정색"에서 "검정"만
  지워져 "색"이 남는 정도의 잔여 노이즈는 허용한다 (임베딩이 견딤).
- 상태는 서버에 저장하지 않는다 — 클라이언트가 병합된 (query, filters)를
  들고 다니는 무상태 설계.
"""
from hybrid import _NEG, FIELD_MAPS, _match, extract_filters


def merge_refinement(prev_query: str, prev_filters: dict, followup: str):
    """(병합 질의 텍스트, 병합 필터) 반환."""
    new = extract_filters(followup)
    neg_words = " ".join(_NEG.findall(followup))
    merged = dict(prev_filters)
    text = prev_query
    for field, mapping in FIELD_MAPS.items():
        if field in new:
            merged[field] = new[field]
            for word in mapping:
                text = text.replace(word, "")
        elif field in merged and _match(neg_words, mapping):
            # 대체 값 없이 부정만 있는 경우 ("네이비 말고") — extract_filters는
            # 긍정 매칭-부정 매칭 상쇄로 빈 dict를 반환해 이 필드를 건드리지 않았을
            # 것이므로, 여기서 별도로 감지해 이전 값을 제거한다. 이게 없으면
            # 부정된 이전 필터가 그대로 남아 검색 결과가 반대로 나간다.
            merged.pop(field, None)
            for word in mapping:
                text = text.replace(word, "")
    followup_clean = _NEG.sub("", followup)
    text = " ".join(f"{text} {followup_clean}".split())
    return text, merged
