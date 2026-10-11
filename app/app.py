from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.cf_model import CFConfig, MeshCollaborativeFiltering
from src.data_utils import extract_terms_from_query, filter_vocabulary, load_csv, query_vector_from_terms

st.set_page_config(page_title="Clinical Trial MeSH Recommender", page_icon="🧬", layout="wide")
st.title("Clinical Trial MeSH Keyword Recommender")
st.caption("Collaborative filtering over MeSH terms — student prototype based on Zhou (CS229, 2020)")


@st.cache_resource
def build_model():
    data_path = ROOT / "data" / "sample_trials.csv"
    df = load_csv(data_path)
    df, vocab = filter_vocabulary(df, min_doc_freq=1, max_terms=100)
    U, vocab = MeshCollaborativeFiltering.tfidf_matrix(df["mesh_terms_list"].tolist())
    model = MeshCollaborativeFiltering(
        CFConfig(
            tau=-5.0,
            k=1.0,
            learning_rate=0.01,
            reg_lambda=0.001,
            epochs=40,
            batch_size=min(64, U.shape[1]),
            seed=42,
        )
    ).fit(U)
    doc_freq = (U > 0).sum(axis=1).astype(np.float32)
    return df, vocab, U, doc_freq, model


try:
    df, vocab, U, doc_freq, model = build_model()
except Exception as exc:
    st.error(f"Unable to build model: {exc}")
    st.stop()

st.info(f"Demo corpus: {len(df)} curated public clinical-trial records | {len(vocab)} MeSH terms")
query = st.text_area(
    "Enter a clinical-trial query",
    value="Clinical study involving Arthritis, Rheumatoid and autoimmune diseases with methotrexate",
    height=110,
)

if st.button("Recommend Keywords", type="primary"):
    matched = extract_terms_from_query(query, vocab)
    vector = query_vector_from_terms(matched, vocab, n_docs=U.shape[1], doc_freq=doc_freq)
    left, right = st.columns(2)
    with left:
        st.subheader("Detected MeSH terms")
        if matched:
            for term in matched:
                st.write(f"- {term}")
        else:
            st.warning("No exact controlled MeSH phrase matched. Try one of the demo queries below.")
    with right:
        st.subheader("Recommended keywords")
        recs = model.recommend(vector, vocab, top_k=5)
        if recs:
            for rank, (term, score) in enumerate(recs, 1):
                st.write(f"**{rank}. {term}**")
                st.caption(f"Predicted score: {score:.4f}")
        else:
            st.info("No recommendation was produced for this query.")
    st.caption("The recommendation is an information-retrieval aid and is not clinical advice.")

with st.expander("Demo queries"):
    st.write("1. Clinical trial involving Arthritis, Rheumatoid and autoimmune diseases with methotrexate")
    st.write("2. Patients with Lung Neoplasms receiving cancer treatment")
    st.write("3. Study of postoperative complications after hernia surgery")

with st.expander("How it works"):
    st.markdown(
        """
        **Query → MeSH matching → TF-IDF vector → similarity-derived collaborative filtering → Top-5 terms**

        The prototype uses a TF-IDF term-document matrix, cosine-similarity-derived neighbor matrix W,
        a ReLU prediction layer, L2 regularization, and mini-batch gradient descent.
        """
    )
