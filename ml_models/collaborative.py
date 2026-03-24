"""KNN user-based collaborative filtering on simulated user–item ratings."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder


def _parse_rating(val: Any) -> float | None:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    s = str(val).strip().lower()
    if "no rating" in s or s in ("", "nan"):
        return None
    try:
        return float(s.split()[0])
    except (ValueError, IndexError):
        return None


def build_dummy_interactions(
    df: pd.DataFrame,
    n_users: int = 800,
    ratings_per_product: tuple[int, int] = (4, 12),
    random_state: int = 42,
    max_products: int | None = 8000,
) -> pd.DataFrame:
    """
    Build user_id, pid, rating interactions when real data has no user history.
    Optionally uses product_rating from the row when parseable.
    """
    rng = np.random.default_rng(random_state)
    work = df
    if max_products is not None and len(work) > max_products:
        work = work.sample(n=max_products, random_state=random_state)

    rows: list[dict] = []
    pid_col = "pid" if "pid" in work.columns else work.columns[0]

    for _, row in work.iterrows():
        pid = str(row[pid_col])
        parsed = _parse_rating(row.get("product_rating"))
        base = parsed if parsed is not None else float(rng.uniform(3.0, 5.0))
        k = int(rng.integers(ratings_per_product[0], ratings_per_product[1] + 1))
        pick = min(k, n_users)
        users = rng.choice(n_users, size=pick, replace=False)
        for u in users:
            noise = float(rng.normal(0, 0.25))
            rating = float(np.clip(base + noise, 1.0, 5.0))
            rows.append({"user_id": int(u), "pid": pid, "rating": rating})

    out = pd.DataFrame(rows)
    out = out.groupby(["user_id", "pid"], as_index=False)["rating"].mean()
    return out


class CollaborativeRecommender:
    """
    User-based KNN in sparse user × item space (cosine on rows).
    """

    def __init__(self, n_neighbors: int = 25) -> None:
        self.n_neighbors = n_neighbors
        self._mat: csr_matrix | None = None
        self._nn: NearestNeighbors | None = None
        self._user_enc = LabelEncoder()
        self._item_enc = LabelEncoder()
        self._global_mean: float = 3.0
        self._item_mean: dict[str, float] = {}

    def fit(self, interactions: pd.DataFrame) -> CollaborativeRecommender:
        if interactions.empty:
            raise ValueError("interactions is empty")
        inter = interactions.groupby(["user_id", "pid"], as_index=False)["rating"].mean()
        self._global_mean = float(inter["rating"].mean())
        self._item_mean = {
            str(k): float(v) for k, v in inter.groupby("pid")["rating"].mean().items()
        }

        u = self._user_enc.fit_transform(inter["user_id"])
        it = self._item_enc.fit_transform(inter["pid"].astype(str))
        data = inter["rating"].values.astype(np.float64)
        n_rows = len(self._user_enc.classes_)
        n_cols = len(self._item_enc.classes_)
        self._mat = csr_matrix((data, (u, it)), shape=(n_rows, n_cols))

        nn = min(self.n_neighbors + 1, max(2, n_rows))
        self._nn = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=nn)
        self._nn.fit(self._mat)
        return self

    def predict_rating(self, user_id: int, pid: str) -> float:
        pid_s = str(pid)
        if self._mat is None or self._nn is None:
            raise RuntimeError("Call fit() first.")
        if pid_s not in set(self._item_enc.classes_):
            return self._global_mean
        try:
            col = int(self._item_enc.transform([pid_s])[0])
        except ValueError:
            return self._global_mean
        try:
            u_row = int(self._user_enc.transform([user_id])[0])
        except ValueError:
            return self._item_mean.get(pid_s, self._global_mean)

        vec = self._mat[u_row]
        dist, ind = self._nn.kneighbors(vec, return_distance=True)
        num, den = 0.0, 0.0
        for d, j in zip(dist[0], ind[0]):
            if j == u_row:
                continue
            r = self._mat[int(j), col]
            if r <= 0:
                continue
            w = max(1e-6, 1.0 - float(d))
            num += w * float(r)
            den += w
        if den <= 0:
            return self._item_mean.get(pid_s, self._global_mean)
        return float(np.clip(num / den, 1.0, 5.0))

    def recommend(self, user_id: int, top_k: int = 10) -> list[tuple[str, float]]:
        """Top predicted ratings among items this user has not rated."""
        if self._mat is None:
            raise RuntimeError("Call fit() first.")
        try:
            u_row = int(self._user_enc.transform([user_id])[0])
        except ValueError:
            top = sorted(self._item_mean.items(), key=lambda x: -x[1])[:top_k]
            return [(p, s) for p, s in top]

        row = self._mat[u_row].toarray().ravel()
        scores: list[tuple[str, float]] = []
        for col in range(self._mat.shape[1]):
            if row[col] > 0:
                continue
            pid = str(self._item_enc.inverse_transform([col])[0])
            scores.append((pid, self.predict_rating(user_id, pid)))
        scores.sort(key=lambda x: -x[1])
        return scores[:top_k]


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    users = np.repeat(np.arange(20), 5)
    items = [f"i{x}" for x in rng.integers(0, 15, size=len(users))]
    ratings = rng.uniform(1, 5, size=len(users))
    inter = pd.DataFrame({"user_id": users, "pid": items, "rating": ratings})
    inter = inter.groupby(["user_id", "pid"], as_index=False)["rating"].mean()

    cf = CollaborativeRecommender(n_neighbors=5)
    cf.fit(inter)
    pr = cf.predict_rating(0, str(inter["pid"].iloc[0]))
    rec = cf.recommend(0, top_k=3)
    print("collaborative test predict_rating:", round(pr, 3), "recommend:", rec)
