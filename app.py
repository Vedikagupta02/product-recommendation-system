from pathlib import Path

import streamlit as st

from recommendation import DEFAULT_DATASET, display_product_recommendation


def main():
    st.title("E-commerce Product Recommendation")
    base = Path(__file__).resolve().parent
    dataset_path = str(base / DEFAULT_DATASET)
    display_product_recommendation(dataset_path)


if __name__ == "__main__":
    main()
