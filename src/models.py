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


_LOADERS = {"clip": ClipEmbedder, "jina": JinaEmbedder}


def load_embedder(model_key: str, device: str | None = None):
    spec = MODELS[model_key]
    return _LOADERS[spec["loader"]](spec["hf_id"], device=device)
