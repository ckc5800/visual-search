"""임베딩 모델 어댑터. 모든 모델은 embed_images / embed_texts 두 메서드로 통일하고
L2 정규화된 float32 (N, D) 배열을 반환한다 — 이후 검색은 내적 = 코사인."""
import numpy as np

from config import MODELS


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norms, 1e-12, None)


def _features_tensor(out):
    # transformers v5는 get_image/text_features가 ModelOutput을 반환한다
    # (pooler_output = projection 적용된 임베딩). v4는 텐서를 그대로 반환.
    return out.pooler_output if hasattr(out, "pooler_output") else out


class ClipEmbedder:
    """transformers CLIPModel 계열 (ko-clip)."""

    def __init__(self, hf_id: str, device: str | None = None):
        import torch
        from transformers import AutoProcessor, CLIPModel

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(hf_id).to(self.device).eval()
        self.processor = AutoProcessor.from_pretrained(hf_id)

    def embed_images(self, images, batch_size: int = 32) -> np.ndarray:
        import torch

        out = []
        for i in range(0, len(images), batch_size):
            batch = self.processor(
                images=images[i : i + batch_size], return_tensors="pt"
            ).to(self.device)
            with torch.no_grad():
                feats = _features_tensor(self.model.get_image_features(**batch))
            out.append(feats.float().cpu().numpy())
        return _l2_normalize(np.concatenate(out))

    def embed_texts(self, texts, batch_size: int = 64) -> np.ndarray:
        import torch

        out = []
        for i in range(0, len(texts), batch_size):
            batch = self.processor(
                text=texts[i : i + batch_size],
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(self.device)
            with torch.no_grad():
                feats = _features_tensor(self.model.get_text_features(**batch))
            out.append(feats.float().cpu().numpy())
        return _l2_normalize(np.concatenate(out))


class JinaEmbedder:
    """jina-clip-v2 — encode_image/encode_text 커스텀 API, trust_remote_code 필요."""

    def __init__(self, hf_id: str, device: str | None = None):
        import torch
        from transformers import AutoModel

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = (
            AutoModel.from_pretrained(hf_id, trust_remote_code=True)
            .to(self.device)
            .eval()
        )

    def embed_images(self, images, batch_size: int = 16) -> np.ndarray:
        emb = self.model.encode_image(images, batch_size=batch_size)
        return _l2_normalize(np.asarray(emb, dtype=np.float32))

    def embed_texts(self, texts, batch_size: int = 32) -> np.ndarray:
        emb = self.model.encode_text(texts, batch_size=batch_size, task="retrieval.query")
        return _l2_normalize(np.asarray(emb, dtype=np.float32))


class MClipEmbedder:
    """멀티링구얼 텍스트 타워 + 영어 CLIP 이미지 타워 (sentence-transformers 두 모델 조합).
    두 타워는 같은 512차원 공간에 정렬돼 있다."""

    def __init__(self, text_hf_id: str, image_hf_id: str, device: str | None = None):
        import torch
        from sentence_transformers import SentenceTransformer

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.text_model = SentenceTransformer(text_hf_id, device=self.device)
        self.image_model = SentenceTransformer(image_hf_id, device=self.device)

    def embed_images(self, images, batch_size: int = 32) -> np.ndarray:
        emb = self.image_model.encode(images, batch_size=batch_size,
                                      convert_to_numpy=True, show_progress_bar=False)
        return _l2_normalize(emb.astype(np.float32))

    def embed_texts(self, texts, batch_size: int = 64) -> np.ndarray:
        emb = self.text_model.encode(texts, batch_size=batch_size,
                                     convert_to_numpy=True, show_progress_bar=False)
        return _l2_normalize(emb.astype(np.float32))


class MtClipEmbedder:
    """번역 파이프라인: 한국어 질의를 MT(ko→en)로 번역한 뒤 영어 CLIP으로 인코딩.
    이미지 타워는 영어 CLIP 그대로 — m-clip과 인덱스를 공유한다.
    번역 결과는 last_translations에 남겨 평가 시 오번역을 추적할 수 있게 한다."""

    def __init__(self, mt_hf_id: str, image_hf_id: str, device: str | None = None,
                 mt_kwargs: dict | None = None):
        import torch
        from sentence_transformers import SentenceTransformer
        from transformers import pipeline

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.translator = pipeline("translation", model=mt_hf_id, device=-1,
                                   **(mt_kwargs or {}))
        self.image_model = SentenceTransformer(image_hf_id, device=self.device)
        self.last_translations: list[str] = []

    def embed_images(self, images, batch_size: int = 32) -> np.ndarray:
        emb = self.image_model.encode(images, batch_size=batch_size,
                                      convert_to_numpy=True, show_progress_bar=False)
        return _l2_normalize(emb.astype(np.float32))

    def embed_texts(self, texts, batch_size: int = 32) -> np.ndarray:
        out = self.translator(list(texts), max_length=64, batch_size=batch_size)
        self.last_translations = [o["translation_text"] for o in out]
        emb = self.image_model.encode(self.last_translations, batch_size=batch_size,
                                      convert_to_numpy=True, show_progress_bar=False)
        return _l2_normalize(emb.astype(np.float32))


# 커머스 용어집: 소형 MT가 반복적으로 틀리는 외래어·콩글리시를 LLM 번역 프롬프트에 주입
COMMERCE_GLOSSARY = {
    "원피스": "dress", "운동화": "sneakers", "힐": "heels", "손목시계": "wristwatch",
    "니트": "knit", "쿠르타": "kurta", "슬리퍼": "flip-flops", "구두": "dress shoes",
    "남아용": "for boys", "여아용": "for girls", "매니큐어": "nail polish",
    "향수": "perfume", "백팩": "backpack", "반팔": "short-sleeve shirt",
    "상의": "top (clothing)", "블라우스": "blouse", "스웨트셔츠": "sweatshirt",
}


class LlmClipEmbedder:
    """LLM(Ollama) 번역 + 영어 CLIP. 질의에 등장한 용어집 항목만 프롬프트에 주입해
    소형 MT의 도메인 오역("운동화→Male Aggression")을 도메인 지식으로 잡는 접근."""

    def __init__(self, ollama_model: str, image_hf_id: str, device: str | None = None):
        import torch
        from sentence_transformers import SentenceTransformer

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ollama_model
        self.image_model = SentenceTransformer(image_hf_id, device=self.device)
        self.last_translations: list[str] = []

    def _translate(self, text: str) -> str:
        import requests

        hits = {k: v for k, v in COMMERCE_GLOSSARY.items() if k in text}
        glossary = "".join(f"\n- {k} = {v}" for k, v in hits.items())
        prompt = (
            "Translate the Korean product-search query into concise English "
            "for an e-commerce image search engine. Output ONLY the English translation, "
            "no quotes, no explanation."
            + (f"\nGlossary (must follow):{glossary}" if glossary else "")
            + f"\nKorean query: {text}\nEnglish:"
        )
        r = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": self.model, "prompt": prompt, "stream": False,
                  "options": {"temperature": 0}},
            timeout=300,
        )
        r.raise_for_status()
        return r.json()["response"].strip().strip('"')

    def embed_images(self, images, batch_size: int = 32) -> np.ndarray:
        emb = self.image_model.encode(images, batch_size=batch_size,
                                      convert_to_numpy=True, show_progress_bar=False)
        return _l2_normalize(emb.astype(np.float32))

    def embed_texts(self, texts, batch_size: int = 32) -> np.ndarray:
        self.last_translations = [self._translate(t) for t in texts]
        emb = self.image_model.encode(self.last_translations, batch_size=batch_size,
                                      convert_to_numpy=True, show_progress_bar=False)
        return _l2_normalize(emb.astype(np.float32))


_LOADERS = {"clip": ClipEmbedder, "jina": JinaEmbedder, "mclip": MClipEmbedder,
            "mtclip": MtClipEmbedder, "llmclip": LlmClipEmbedder}


def load_embedder(model_key: str, device: str | None = None):
    spec = MODELS[model_key]
    cls = _LOADERS[spec["loader"]]
    kwargs = {}
    if "image_hf_id" in spec:
        kwargs["image_hf_id"] = spec["image_hf_id"]
    if "mt_kwargs" in spec:
        kwargs["mt_kwargs"] = spec["mt_kwargs"]
    return cls(spec["hf_id"], device=device, **kwargs)
