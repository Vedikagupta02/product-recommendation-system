import os
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from data_processing import preprocess_data
from ml_models.collaborative import CollaborativeRecommender, build_dummy_interactions
from ml_models.content_based import ContentBasedRecommender
from ml_models.hybrid import format_products_for_prompt, get_hybrid_recommendations

load_dotenv()

DEFAULT_OLLAMA_MODEL = "llama3.2"
DEFAULT_DATASET = "flipkart_com-ecommerce_sample.csv"


@st.cache_resource(show_spinner="Training hybrid recommenders (one-time)…")
def load_hybrid_stack(dataset_path: str):
    """Load CSV, preprocess, fit content-based + collaborative models."""
    path = Path(dataset_path)
    if not path.is_file():
        path = Path(__file__).resolve().parent / dataset_path
    df = pd.read_csv(path, on_bad_lines="skip", engine="python")
    refined = preprocess_data(df)
    content = ContentBasedRecommender()
    content.fit(refined)
    inter = build_dummy_interactions(refined)
    collab = CollaborativeRecommender()
    collab.fit(inter)
    return refined, content, collab


__all__ = [
    "load_hybrid_stack",
    "display_product_recommendation",
    "get_hybrid_recommendations",
    "format_products_for_prompt",
    "DEFAULT_DATASET",
]


def display_product_recommendation(dataset_path: str = DEFAULT_DATASET):
    """
    Hybrid ML (content → collaborative re-rank) + Ollama explanation layer.
    """
    st.header("Product Recommendation")

    with st.expander("Setup (free, runs on your PC)", expanded=False):
        st.markdown(
            """
            1. Install **[Ollama](https://ollama.com)** and start it.
            2. Run: `ollama pull llama3.2` (or set `OLLAMA_MODEL` in `.env`).
            3. **Hybrid flow:** TF-IDF finds 20 candidates → collaborative re-ranks if you enter a **User ID** → Ollama explains the final list.

            No OpenAI API key required.
            """
        )

    try:
        refined, content, collab = load_hybrid_stack(dataset_path)
    except Exception as e:
        st.error(f"Could not load dataset or train models: {e}")
        return

    model_name = os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
    llm = ChatOllama(model=model_name, temperature=0)

    rag_prompt = ChatPromptTemplate.from_template(
        """You are a concise retail assistant. These products were shortlisted by a hybrid system (content similarity + optional collaborative filtering).

{context}

Shopper preferences:
- Department: {department}
- Category: {category}
- Brand: {brand}
- Max price: {price}
- User ID (for collaborative step): {user_label}

In 2–4 short paragraphs: (1) why these items fit the query, (2) mention the top 3 by name and why they rank first. If the list is empty, say you found no close matches."""
    )
    chain = rag_prompt | llm

    department = st.text_input("Product Department")
    category = st.text_input("Product Category")
    brand = st.text_input("Product Brand")
    price = st.text_input("Maximum Price Range")
    user_raw = st.text_input(
        "User ID (optional, 0–799 for simulated users — enables collaborative re-ranking)",
        value="",
    )

    if st.button("Get Recommendations"):
        query = " ".join(
            x for x in [department, category, brand, price] if str(x).strip()
        )
        if not query.strip():
            st.warning("Enter at least one preference field to build a search query.")
            return

        user_id = None
        if str(user_raw).strip() != "":
            try:
                user_id = int(user_raw)
            except ValueError:
                st.warning("User ID must be an integer; skipping collaborative re-rank.")
                user_id = None

        try:
            with st.spinner("Running hybrid retrieval…"):
                hybrid = get_hybrid_recommendations(
                    query=query,
                    refined_df=refined,
                    content=content,
                    collab=collab,
                    user_id=user_id,
                    content_top_k=20,
                    final_top_k=10,
                )
            ctx = format_products_for_prompt(hybrid["rows"])
            user_label = str(user_id) if user_id is not None else "not set (content-only order)"

            with st.spinner("Generating explanation with Ollama…"):
                result = chain.invoke(
                    {
                        "context": ctx or "(No products returned — try broader keywords.)",
                        "department": department or "—",
                        "category": category or "—",
                        "brand": brand or "—",
                        "price": price or "—",
                        "user_label": user_label,
                    }
                )
            text = getattr(result, "content", None) or str(result)
            st.subheader("Explanation (RAG)")
            st.write(text)
            if not hybrid["rows"].empty:
                st.subheader("Hybrid shortlist (ML)")
                st.dataframe(
                    hybrid["rows"][
                        [
                            c
                            for c in [
                                "pid",
                                "product_name",
                                "brand",
                                "primary_category",
                                "discounted_price",
                                "retail_price",
                            ]
                            if c in hybrid["rows"].columns
                        ]
                    ],
                    use_container_width=True,
                )
        except Exception as e:
            st.error(
                f"Pipeline error (check Ollama is running: `ollama serve`). Details: {e}"
            )
