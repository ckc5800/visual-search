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

    def __init__(self, mt_hf_id: str, image_hf_id: str, device: str | None = None):
        import torch
        from sentence_transformers import SentenceTransformer
        from transformers import pipeline

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.translator = pipeline("translation", model=mt_hf_id, device=-1)
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


_LOADERS = {"clip": ClipEmbedder, "jina": JinaEmbedder, "mclip": MClipEmbedder,
            "mtclip": MtClipEmbedder}


def load_embedder(model_key: str, device: str | None = None):
    spec = MODELS[model_key]
    cls = _LOADERS[spec["loader"]]
    if "image_hf_id" in spec:
        return cls(spec["hf_id"], spec["image_hf_id"], device=device)
    return cls(spec["hf_id"], device=device)
