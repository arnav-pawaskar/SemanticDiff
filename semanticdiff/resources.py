"""Lazily loaded, process-wide NLP resources (spaCy, sentence encoder, NLI cross-encoder).

Models are loaded once and cached; the Streamlit app additionally wraps these in
``st.cache_resource``. Nothing here downloads data at import time.
"""
from __future__ import annotations

import functools
import logging
from typing import Any

log = logging.getLogger(__name__)


def resolve_device(preference: str = "auto") -> str:
    if preference in ("cuda", "cpu"):
        return preference
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:  # pragma: no cover
        return "cpu"


@functools.lru_cache(maxsize=4)
def get_spacy(name: str = "en_core_web_lg"):
    import spacy
    try:
        nlp = spacy.load(name)
    except OSError:
        log.warning("spaCy model %s not found, falling back to en_core_web_sm", name)
        nlp = spacy.load("en_core_web_sm")
    return nlp


@functools.lru_cache(maxsize=4)
def get_encoder(name: str, device: str = "auto"):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(name, device=resolve_device(device))


@functools.lru_cache(maxsize=4)
def get_nli(name: str, device: str = "auto"):
    """Return ``(CrossEncoder, label_index)`` where label_index maps
    'entailment' / 'contradiction' / 'neutral' to output column indices,
    read from the model config rather than assumed."""
    from sentence_transformers import CrossEncoder
    model = CrossEncoder(name, device=resolve_device(device))
    id2label: dict[int, str] = model.config.id2label
    label_index = {label.lower(): int(idx) for idx, label in id2label.items()}
    missing = {"entailment", "contradiction", "neutral"} - label_index.keys()
    if missing:
        raise ValueError(f"NLI model {name} lacks labels {missing}: {id2label}")
    return model, label_index


def get_all(cfg: dict[str, Any]):
    m = cfg["models"]
    dev = cfg.get("device", "auto")
    return get_spacy(m["spacy"]), get_encoder(m["embedding"], dev), get_nli(m["nli"], dev)
