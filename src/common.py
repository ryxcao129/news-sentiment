"""Shared paths and helpers for the news sentiment project."""
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
for p in (DATA, MODELS, REPORTS):
    p.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Label order used everywhere (matches ProsusAI/finbert's output head)
LABELS = ["positive", "negative", "neutral"]
BASE_MODEL = "ProsusAI/finbert"
FT_DIR = MODELS / "finbert_finetuned"


def sentiment_score(probs: np.ndarray) -> np.ndarray:
    """Collapse 3-class probabilities to one number in [-1, 1]: P(positive) - P(negative)."""
    return probs[:, 0] - probs[:, 1]


@torch.no_grad()
def predict_probs(texts, model, tokenizer, batch=64, max_len=64):
    model.eval().to(DEVICE)
    out = []
    for i in range(0, len(texts), batch):
        enc = tokenizer(list(texts[i:i + batch]), padding=True, truncation=True,
                        max_length=max_len, return_tensors="pt").to(DEVICE)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=DEVICE == "cuda"):
            logits = model(**enc).logits
        out.append(torch.softmax(logits.float(), dim=-1).cpu().numpy())
    return np.vstack(out) if out else np.zeros((0, 3))
