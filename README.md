# visual-search — 한국어 멀티모달 상품 검색

[![ci](https://github.com/ckc5800/visual-search/actions/workflows/ci.yml/badge.svg)](https://github.com/ckc5800/visual-search/actions/workflows/ci.yml)

한국어 문장형 질의("체크무늬 네이비 남성 셔츠")로 상품 이미지를 검색하는 프로젝트.
텍스트 RAG([portfolio-rag-agent](https://github.com/ckc5800/portfolio-rag-agent))에서 쓴
검색 평가 방법론(recall@k·MRR, 정직한 실패 기록)을 멀티모달로 확장하는 것이 목표다.

## 구조

- 데이터: [ashraq/fashion-product-images-small](https://huggingface.co/datasets/ashraq/fashion-product-images-small)
  — 상품 44.1K, 이미지(60×80) + 구조화 메타데이터(articleType, baseColour, gender, season, usage 등)
- 임베딩 모델 (A/B 비교 대상, `--model`로 선택):
  - `ko-clip` (기본): [Bingsu/clip-vit-base-patch32-ko](https://huggingface.co/Bingsu/clip-vit-base-patch32-ko) — 한국어 특화 CLIP, 151M, MIT
  - `jina-v2`: [jinaai/jina-clip-v2](https://huggingface.co/jinaai/jina-clip-v2) — 멀티링구얼 89개 언어, 865M, **CC-BY-NC 4.0(비상업)** — 본 저장소는 비상업 포트폴리오 용도로만 사용
- 검색: 임베딩 L2 정규화 후 numpy 내적(=코사인) 정확 탐색.
  44K 규모에서는 ANN이 불필요하다고 판단 — FAISS 등은 규모가 커질 때 도입.

## 사용

```bash
pip install -r requirements.txt

# 인덱스 구축 (스모크: --limit 200 은 스트리밍이라 전체 다운로드 없음)
python src/ingest.py --limit 200
python src/ingest.py                # 전체 44K

# 검색
python src/search.py "체크무늬 네이비 남성 셔츠"
python src/search.py --model jina-v2 --top-k 10 "여름용 캐주얼 원피스"
python src/search.py --image my_shirt.jpg     # 이미지로 비슷한 상품 찾기

# 데모 웹 UI (질의 → 상품 이미지 그리드, 이미지 클릭 → 비슷한 상품)
python -m uvicorn --app-dir src api:app --port 8321
```

이미지 질의 주의: 이 데이터셋은 60×80 저해상도라 JPEG 재인코딩(quality 75)만으로도
자기 자신과의 유사도가 1.000 → 0.912로 떨어진다(근사 중복 상품에 밀릴 수 있음).
무손실(PNG) 또는 원본 픽셀로 질의하면 자기 자신이 정확히 1위로 나오는 것 확인함.

## 평가 결과

한국어 50문항 × 전체 인덱스 44,072개, gold는 메타데이터 필터 기반(`eval/queries_ko.jsonl`).
측정: 2026-08-02, `python src/evaluate.py`, i7-4790 CPU. 재현 메타(모델 id·시각·top-10 id)는
`eval/results_ko-clip.json`의 `run`에 기록.

| model | hit@1 | hit@3 | hit@5 | hit@10 | precision@10 | MRR |
|---|---|---|---|---|---|---|
| ko-clip (임베딩 단독) | 0.720 | 0.820 | 0.860 | 0.900 | 0.652 | 0.780 |
| ko-clip + 하이브리드 필터 | **0.800** | 0.900 | 0.920 | **0.980** | **0.786** | 0.856 |

**주의 — hit@k는 gold가 넓은 질의에서 후하게 나온다.** "남성 셔츠"처럼 gold가 수천 개면
top-10 안에 하나 들어가는 게 거의 자동이다. 그래서 gold 크기 구간별로 분해해 함께 본다:

| gold 크기 | 문항 | hit@1 | hit@10 | precision@10 | MRR |
|---|---|---|---|---|---|
| ≤50 (세부 조건) | 3 | 0.33 | 0.67 | **0.10** | 0.39 |
| 51~500 | 27 | 0.78 | 0.89 | 0.64 | 0.82 |
| 500+ (넓은 카테고리) | 20 | 0.70 | 0.95 | 0.75 | 0.78 |

조건이 좁아질수록 임베딩 단독 검색의 정밀도가 급락한다 — 색·성별 같은 구조화 조건을
메타 필터로 결합해야 한다는(로드맵 3) 실측 근거.

### 하이브리드 필터 (`--hybrid`, `src/hybrid.py`)

질의에서 색·성별·용도를 룰 기반(사전 매핑)으로 추출해 메타 필터로 걸고, 임베딩은
필터 통과분 안에서만 순위를 매긴다. 카테고리는 임베딩이 이미 잘 맞혀서 추출하지 않는다.
필터 통과분이 0이면 필터 무시(폴백), k 미만이면 그만큼만 반환(precision 분모는 k 고정).

- 세부 조건 구간(gold≤50)에서 precision@10 **0.10 → 0.53**, hit@10 0.67 → 1.00 —
  "좁은 조건은 임베딩 단독으로 안 된다"던 위 진단이 필터 결합으로 실제 해소됨을 확인.
- 하이브리드 후 남은 실패는 "인도 전통 여성 쿠르타" 1건 — 색·성별 필터로는 못 푸는
  어휘 이해 문제 (멀티링구얼 모델 비교의 관전 포인트).
- 공정성 주의: gold 정의와 필터가 같은 메타데이터 어휘를 공유하므로, 이 수치는
  "추출이 정확할 때"의 상한에 가깝다. 실서비스라면 추출 오류(동음이의어, 부정어 등)가
  추가 감점 요인이 된다.

hit@10 실패 5건 분석 (임베딩 단독 기준):

- **"인도 전통 여성 쿠르타"** — gold가 1,993개나 되는데 전부 놓침. 외래어 "쿠르타"를
  ko-clip이 이해하지 못하는 것으로 추정 (어휘 밖 단어에 취약).
- **"실버 여성 손목시계"(gold 129)·"분홍색 여성 샌들"(51)·"남아용 신발"(27)·
  "스포츠용 백팩"(83)** — 카테고리는 맞히지만 색상·성별·용도 세부 조건에서 어긋남.
- 구 "겨울 스웨터" 문항은 Sweaters의 season 라벨 87%가 Fall이라 gold 13개짜리
  메타데이터 함정이었음 — "니트 스웨터"로 교체(파일에 사유 주석). 질의 의도가
  메타데이터로 표현 안 되는 문항 4건은 `note`로 한계를 명시.

## 로드맵

1. ~~베이스라인~~: 인제스트 → 한국어 질의 top-k 검색 (완료)
2. **(현재) 평가셋**: 한국어 질의 50문항, 메타데이터 기반 gold 라벨 — ko-clip 측정 완료,
   jina-v2 / 영어 CLIP+질의 번역 비교 예정
3. **하이브리드 필터**: 질의에서 구조화 조건(색·카테고리) 추출 → 메타 필터 + 벡터 검색 결합
4. **대화형 정교화**: "좀 더 캐주얼하게" 류 후속 발화 처리
5. **이미지 질의**: 비슷한 상품 찾기 (image-to-image)

측정된 수치는 실측 시점의 명령·조건과 함께 이 README에 기록한다.
