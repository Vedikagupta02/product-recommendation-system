"""Hybrid ML stages + handoff to RAG (Ollama) for explanations."""

from __future__ import annotations

import pandas as pd

from .collaborative import CollaborativeRecommender
from .content_based import ContentBasedRecommender


def get_hybrid_recommendations(
    query: str,
    refined_df: pd.DataFrame,
    content: ContentBasedRecommender,
    collab: CollaborativeRecommender,
    user_id: int | None = None,
    content_top_k: int = 20,
    final_top_k: int = 10,
    collab_weight: float = 0.45,
) -> dict:
    """
    Step 1: content-based retrieves `content_top_k` candidates (cosine TF-IDF).
    Step 2: if `user_id` is set, re-rank by blending normalized content score and
            collaborative predicted rating; otherwise keep content order.
    Step 3: return ordered product rows for the RAG layer to narrate.

    Returns keys: pids, scores, rows (DataFrame aligned to final order).
    """
    cands = content.recommend(query, top_k=content_top_k)
    if not cands:
        return {"pids": [], "scores": [], "rows": refined_df.iloc[:0].copy()}

    pids_order = [p for p, _ in cands]
    content_scores = {p: float(s) for p, s in cands}

    if user_id is not None:
        blended: list[tuple[str, float]] = []
        for pid in pids_order:
            cs = content_scores[pid]
            pr = collab.predict_rating(user_id, pid)
            norm_collab = pr / 5.0
            score = (1.0 - collab_weight) * cs + collab_weight * norm_collab
            blended.append((pid, score))
        blended.sort(key=lambda x: -x[1])
        top = blended[:final_top_k]
        out_pids = [p for p, _ in top]
        out_scores = [s for _, s in top]
    else:
        out_pids = pids_order[:final_top_k]
        out_scores = [content_scores[p] for p in out_pids]

    order = {pid: i for i, pid in enumerate(out_pids)}
    sub = refined_df[refined_df["pid"].astype(str).isin(out_pids)].copy()
    sub["_ord"] = sub["pid"].astype(str).map(order)
    sub = sub.sort_values("_ord").drop(columns=["_ord"])

    return {"pids": out_pids, "scores": out_scores, "rows": sub}


def format_products_for_prompt(rows: pd.DataFrame, max_chars: int = 6000) -> str:
    """Compact bullet list of catalog rows for the LLM."""
    lines: list[str] = []
    for _, r in rows.iterrows():
        name = str(r.get("product_name", ""))[:200]
        brand = str(r.get("brand", ""))
        cat = str(r.get("primary_category", ""))
        price = r.get("discounted_price", "")
        pid = str(r.get("pid", ""))
        lines.append(f"- [{pid}] {name} | {brand} | {cat} | price {price}")
    text = "\n".join(lines)
    if len(text) > max_chars:
        return text[: max_chars - 3] + "..."
    return text


if __name__ == "__main__":
    import numpy as np
    import pandas as pd

    from .collaborative import CollaborativeRecommender, build_dummy_interactions
    from .content_based import ContentBasedRecommender

    rng = np.random.default_rng(1)
    tiny = pd.DataFrame(
        {
            "pid": [f"P{i}" for i in range(30)],
            "product_url": [""] * 30,
            "product_name": [f"item {i} cotton shirt" for i in range(30)],
            "primary_category": rng.choice(["Clothing", "Shoes"], size=30),
            "retail_price": rng.integers(500, 2000, size=30),
            "discounted_price": rng.integers(300, 1500, size=30),
            "primary_image_link": [""] * 30,
            "description": ["soft fabric"] * 30,
            "brand": [f"B{i % 5}" for i in range(30)],
            "gender": ["Unisex"] * 30,
        }
    )
    cb = ContentBasedRecommender(max_features=200, min_df=1)
    cb.fit(tiny)
    inter = build_dummy_interactions(tiny, n_users=50, max_products=30)
    cf = CollaborativeRecommender(n_neighbors=5)
    cf.fit(inter)
    out = get_hybrid_recommendations(
        "cotton shirt clothing",
        tiny,
        cb,
        cf,
        user_id=3,
        content_top_k=15,
        final_top_k=5,
    )
    print("hybrid test pids:", out["pids"])
    print(format_products_for_prompt(out["rows"])[:400])
