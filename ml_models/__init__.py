"""Lightweight hybrid recommendation models (content-based + collaborative)."""

from .collaborative import CollaborativeRecommender, build_dummy_interactions
from .content_based import ContentBasedRecommender
from .hybrid import format_products_for_prompt, get_hybrid_recommendations

__all__ = [
    "ContentBasedRecommender",
    "CollaborativeRecommender",
    "build_dummy_interactions",
    "get_hybrid_recommendations",
    "format_products_for_prompt",
]
