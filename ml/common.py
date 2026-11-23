"""Shared data helpers for the ticket classifier."""
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse.csgraph import connected_components
from scipy.sparse import coo_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parent.parent
DATA_CSV = ROOT / "data" / "bitext.csv"
MODEL_PATH = ROOT / "ml" / "artifacts" / "classifier.joblib"
# Cosine similarity on char n-grams. 0.85 groups near-duplicates tightly without chaining into one giant
# group (at 0.80 a 2,810-row group forms and leaves CANCEL/SUBSCRIPTION with ~20 test rows).
NEAR_DUP_SIM = 0.85


def normalize(text: str) -> str:
    """Lowercase, remove {{placeholders}}, drop punctuation, squeeze spaces.

    (Placeholders are removed, not kept: the 'X' they are first replaced with is uppercase and is then
    stripped by the character filter. The counts in docs/b1_report.md were computed with this behaviour.)
    """
    t = re.sub(r"\{\{.*?\}\}", "X", str(text).lower())
    t = re.sub(r"[^a-z0-9 ]", "", t)
    return re.sub(r"\s+", " ", t).strip()


def load_raw() -> pd.DataFrame:
    df = pd.read_csv(DATA_CSV)
    df["norm"] = df["instruction"].map(normalize)
    return df


def dedupe(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows whose normalized text already appeared (no label conflicts exist)."""
    return df.drop_duplicates("norm").reset_index(drop=True)


def near_dup_groups(norms: pd.Series, threshold: float = NEAR_DUP_SIM, k: int = 10) -> np.ndarray:
    """Group texts into connected components of the near-duplicate graph.

    Every pair with char-ngram cosine similarity >= threshold ends up in the same
    group, so a whole group can be put on one side of a split.
    """
    X = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5)).fit_transform(norms)
    nn = NearestNeighbors(n_neighbors=k, metric="cosine").fit(X)
    dist, idx = nn.kneighbors(X)
    rows, cols = [], []
    for i in range(len(norms)):
        # Skip self by index, NOT by position: with exact-tie vectors the row itself is not always first.
        for d, j in zip(dist[i], idx[i]):
            if j != i and 1 - d >= threshold:
                rows.append(i)
                cols.append(j)
    n = len(norms)
    graph = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    _, labels = connected_components(graph, directed=False)
    return labels
