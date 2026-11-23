"""Loads the trained category model and a simple rule-based priority."""
import re
from functools import lru_cache
from typing import Tuple

import joblib

from . import config

URGENT = re.compile(
    r"\b(urgent|asap|immediately|fraud|unauthori[sz]ed|hacked|stolen|charged twice|double[- ]charged?|"
    r"locked out|can'?t log ?in|cannot log ?in|scam)\b",
    re.I,
)
LOW_CATEGORIES = {"FEEDBACK", "SUBSCRIPTION", "CONTACT"}
HIGH_CATEGORIES = {"PAYMENT", "REFUND"}


@lru_cache(maxsize=1)
def _model():
    return joblib.load(config.MODEL_PATH)


def classify(text: str) -> Tuple[str, float]:
    """Return (category, probability of that category)."""
    model = _model()
    probs = model.predict_proba([text])[0]
    i = int(probs.argmax())
    return str(model.classes_[i]), float(probs[i])


def priority(category: str, text: str) -> str:
    """Transparent heuristic: urgency words or money categories raise it, chatty categories lower it."""
    if URGENT.search(text):
        return "high"
    if category in HIGH_CATEGORIES:
        return "medium"
    if category in LOW_CATEGORIES:
        return "low"
    return "medium"
