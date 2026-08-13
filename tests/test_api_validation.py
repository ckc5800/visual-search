"""API 입력 검증 회귀 테스트 — 모델·인덱스 없이 검증 계층만 돌린다.

**왜 필요한가 (실제로 있던 결함)**

  1. `/api/refine` 이 클라이언트가 들고 다니는 `filters`(사용자 입력)를
     `json.loads` 에 그대로 넣어, 깨진 JSON 이 500 으로 나갔다.
  2. `alpha` 에 경계가 없어 `[0,1]` 밖 값이 조용히 통과했다. 합성 벡터를
     다시 정규화하기 때문에 결과가 "그럴듯하게" 나와서 틀린 줄도 모른다 —
     alpha=5 는 텍스트를 음수 가중으로 빼는 셈이다.
  3. `q`·`k` 에 상한이 없어 임의 길이 문자열이 임베딩 모델로 갔다.

FastAPI 의 `Query(...)` 제약은 **핸들러 본문에 들어가기 전에** 422 로
끊으므로, STATE 가 비어 있어도(=모델 미적재) 검증만 따로 검사할 수 있다.
그래서 이 테스트는 CI 에서 모델 다운로드 없이 돈다 — 경계를 벗어난 요청이
핸들러에 도달하지 않는다는 것 자체가 검증 대상이다.
"""
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import api  # noqa: E402


@pytest.fixture(scope="module")
def client():
    # lifespan 을 돌리지 않는다 — 모델·데이터셋 적재 없이 검증 계층만 본다.
    # raise_server_exceptions=False 로 둬야 핸들러 본문이 STATE 없음으로
    # 터질 때 예외가 아니라 500 응답으로 돌아온다. 그래야 "검증은 통과했고
    # 본문에서 죽었다"(=500)와 "검증에서 끊겼다"(=422)를 구분할 수 있다.
    return TestClient(api.app, raise_server_exceptions=False)


@pytest.mark.parametrize("params", [
    {"q": "", "k": 5},                                  # 빈 질의
    {"q": "가" * (api.MAX_QUERY_CHARS + 1)},            # 길이 상한 초과
    {"q": "셔츠", "k": 0},                              # k 하한
    {"q": "셔츠", "k": api.MAX_K + 1},                  # k 상한
])
def test_search_rejects_out_of_range(client, params):
    assert client.get("/api/search", params=params).status_code == 422


@pytest.mark.parametrize("alpha", [-3.0, -0.01, 1.01, 5.0])
def test_compose_rejects_alpha_outside_unit_interval(client, alpha):
    """검증 전에는 이 값들이 통과해 엉뚱한 합성 벡터를 만들었다."""
    r = client.get("/api/compose", params={"pos": 0, "text": "검정색", "alpha": alpha})
    assert r.status_code == 422


def test_compose_accepts_alpha_endpoints(client):
    """0.0·1.0 은 정상 범위다 — 경계를 너무 좁게 잡지 않았는지 고정한다.

    STATE 가 비어 있어 핸들러 본문은 실패하지만, 그건 **검증을 통과했다**는
    뜻이다. 422(검증 거부)만 아니면 된다.
    """
    for alpha in (0.0, 1.0):
        r = client.get("/api/compose", params={"pos": 0, "text": "검정색", "alpha": alpha})
        assert r.status_code != 422


def test_refine_returns_422_not_500_on_broken_filters(client):
    """깨진 JSON 은 서버 오류가 아니라 잘못된 요청이다."""
    r = client.get("/api/refine",
                   params={"q": "셔츠", "followup": "검정색으로", "filters": "{bad"})
    assert r.status_code == 422
    assert "JSON" in r.text


def test_refine_rejects_non_object_filters(client):
    """filters 는 객체여야 한다 — 배열·스칼라는 merge_refinement 가 못 받는다."""
    for bad in ("[1,2]", '"문자열"', "42"):
        r = client.get("/api/refine",
                       params={"q": "셔츠", "followup": "검정색으로", "filters": bad})
        assert r.status_code == 422


def test_similar_bounds_k(client):
    assert client.get("/api/similar", params={"pos": 0, "k": 0}).status_code == 422
    assert client.get("/api/similar",
                      params={"pos": 0, "k": api.MAX_K + 1}).status_code == 422
