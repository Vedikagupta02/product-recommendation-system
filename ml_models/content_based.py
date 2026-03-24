"""TF-IDF content-based product similarity."""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def _clean_text(x: Any) -> str:
    if pd.isna(x) or x is None:
        return ""
    s = str(x).lower()
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def _combined_text(row: pd.Series) -> str:
    parts = [
        _clean_text(row.get("product_name", "")),
        _clean_text(row.get("description", "")),
        _clean_text(row.get("primary_category", "")),
        _clean_text(row.get("brand", "")),
    ]
    return " ".join(p for p in parts if p)


class ContentBasedRecommender:
    """Recommend products by cosine similarity of TF-IDF vectors."""

    def __init__(
        self,
        max_features: int = 8000,
        ngram_range: tuple[int, int] = (1, 2),
        min_df: int = 2,
    ) -> None:
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            stop_words="english",
        )
        self._matrix = None
        self._df: pd.DataFrame | None = None
        self._pid_to_row: dict[str, int] = {}

    def fit(self, df: pd.DataFrame) -> ContentBasedRecommender:
        """Fit on a catalog with columns: pid, product_name, description, primary_category, brand."""
        self._df = df.reset_index(drop=True).copy()
        corpus = self._df.apply(_combined_text, axis=1).tolist()
        self._matrix = self.vectorizer.fit_transform(corpus)
        self._pid_to_row = {str(p): i for i, p in enumerate(self._df["pid"].astype(str))}
        return self

    def recommend(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        """
        Return top_k (pid, cosine_similarity) pairs for a free-text query.
        """
        if self._matrix is None or self._df is None:
            raise RuntimeError("Call fit() before recommend().")
        q = self.vectorizer.transform([_clean_text(query)])
        sims = cosine_similarity(q, self._matrix).ravel()
        k = min(top_k, len(sims))
        top_idx = np.argpartition(sims, -k)[-k:]
        top_idx = top_idx[np.argsort(sims[top_idx])[::-1]]
        pids = self._df["pid"].astype(str).values
        return [(str(pids[i]), float(sims[i])) for i in top_idx]


if __name__ == "__main__":
    # Tiny smoke test
    sample = pd.DataFrame(
        {
            "pid": ["p1", "p2", "p3"],
            "product_name": ["Blue cotton shirt", "Red running shoes", "Cotton casual shirt"],
            "description": ["comfortable cotton", "lightweight sole", "soft cotton fabric"],
            "primary_category": ["Clothing", "Footwear", "Clothing"],
            "brand": ["A", "B", "A"],
        }
    )
    cb = ContentBasedRecommender(max_features=100, min_df=1)
    cb.fit(sample)
    out = cb.recommend("cotton shirt clothing", top_k=2)
    print("content_based test:", out)
