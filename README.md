# visual-search — 한국어 멀티모달 상품 검색

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
```

## 평가 결과

한국어 50문항 × 전체 인덱스 44,072개, gold는 메타데이터 필터 기반(`eval/queries_ko.jsonl`).
측정: 2026-08-02, `python src/evaluate.py`, i7-4790 CPU.

| model | hit@1 | hit@3 | hit@5 | hit@10 | MRR |
|---|---|---|---|---|---|
| ko-clip | 0.720 | 0.820 | 0.860 | 0.900 | 0.780 |

hit@10 실패 5건 분석 (`eval/results_ko-clip.json`에 문항별 기록):

- **"인도 전통 여성 쿠르타"** — gold가 1,993개나 되는데 전부 놓침. 외래어 "쿠르타"를
  ko-clip이 이해하지 못하는 것으로 추정 (어휘 밖 단어에 취약).
- **"실버 여성 손목시계"(gold 129)·"분홍색 여성 샌들"(51)·"남아용 신발"(27)** —
  카테고리는 맞히지만 색상·성별 세부 조건에서 어긋남. 색·성별 같은 구조화 조건은
  임베딩 단독보다 메타 필터 결합(로드맵 3)이 맞는 방향이라는 근거.
- **"겨울 스웨터"(gold 13)** — gold 자체가 13개뿐인 희소 문항.

## 로드맵

1. ~~베이스라인~~: 인제스트 → 한국어 질의 top-k 검색 (완료)
2. **(현재) 평가셋**: 한국어 질의 50문항, 메타데이터 기반 gold 라벨 — ko-clip 측정 완료,
   jina-v2 / 영어 CLIP+질의 번역 비교 예정
3. **하이브리드 필터**: 질의에서 구조화 조건(색·카테고리) 추출 → 메타 필터 + 벡터 검색 결합
4. **대화형 정교화**: "좀 더 캐주얼하게" 류 후속 발화 처리
5. **이미지 질의**: 비슷한 상품 찾기 (image-to-image)

측정된 수치는 실측 시점의 명령·조건과 함께 이 README에 기록한다.
