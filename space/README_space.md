---
title: visual-search — 한국어 상품 검색
emoji: 🔍
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# visual-search — 한국어 멀티모달 상품 검색 데모

한국어 문장형 질의("체크무늬 네이비 남성 셔츠")로 패션 상품 44,072개를 검색합니다.

- 임베딩: [Bingsu/clip-vit-base-patch32-ko](https://huggingface.co/Bingsu/clip-vit-base-patch32-ko) (한국어 CLIP)
- 하이브리드: 질의에서 색·성별·용도를 추출해 메타 필터 결합 (hit@10 90%→98%)
- 대화형 정교화: "좀 더 캐주얼하게", "네이비 말고 검정색으로"
- 데이터: [ashraq/fashion-product-images-small](https://huggingface.co/datasets/ashraq/fashion-product-images-small)

코드·평가 리포트: https://github.com/ckc5800/visual-search
