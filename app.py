import os
import re
import faiss
import pickle
import fitz
import numpy as np
import streamlit as st

from rank_bm25 import BM25Okapi

from sentence_transformers import (
    SentenceTransformer,
    CrossEncoder
)

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# =====================================================
# CONFIG
# =====================================================

st.set_page_config(
    page_title="Hybrid Search STKI",
    layout="wide"
)

# =====================================================
# PREPROCESSING
# =====================================================

stemmer = StemmerFactory().create_stemmer()


def preprocessing_teks(teks):

    teks = teks.lower()

    teks = re.sub(
        r'[^a-zA-Z\s]',
        ' ',
        teks
    )

    tokens = teks.split()

    hasil = []

    for token in tokens:

        if len(token) <= 2:
            continue

        hasil.append(
            stemmer.stem(token)
        )

    return hasil


# =====================================================
# LOAD PDF
# =====================================================

@st.cache_resource
def load_documents():

    folders = [
        "TOPIK_1_UMKM",
        "TOPIK_2_KETENAGAKERJAAN"
    ]

    documents = []
    doc_ids = []

    for folder in folders:

        for file_name in os.listdir(folder):

            if file_name.endswith(".pdf"):

                path_pdf = os.path.join(
                    folder,
                    file_name
                )

                teks = ""

                try:

                    doc = fitz.open(path_pdf)

                    for page in doc:

                        teks += (
                            page.get_text()
                            + " "
                        )

                    documents.append(teks)

                    doc_ids.append(
                        file_name.replace(
                            ".pdf",
                            ""
                        )
                    )

                except Exception:

                    pass

    return doc_ids, documents


# =====================================================
# LOAD DATA
# =====================================================

with st.spinner("Loading Documents..."):

    doc_ids, documents = load_documents()

doc_dict = {

    doc_id: doc

    for doc_id, doc
    in zip(doc_ids, documents)
}

# =====================================================
# BM25
# =====================================================

@st.cache_resource
def build_bm25():

    tokenized_corpus = [

        preprocessing_teks(doc)

        for doc in documents
    ]

    return BM25Okapi(
        tokenized_corpus
    )

with st.spinner("Building BM25..."):

    bm25 = build_bm25()

# =====================================================
# SBERT
# =====================================================

@st.cache_resource
def load_sbert():

    return SentenceTransformer(
        "paraphrase-multilingual-MiniLM-L12-v2"
    )

with st.spinner("Loading Semantic Model..."):

    model = load_sbert()

# =====================================================
# CROSS ENCODER
# =====================================================

@st.cache_resource
def load_cross_encoder():

    return CrossEncoder(
        "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )

with st.spinner("Loading Cross Encoder..."):

    cross_encoder = load_cross_encoder()

# =====================================================
# FAISS
# =====================================================

@st.cache_resource
def load_faiss():

    index = faiss.read_index(
        "index_semantic.faiss"
    )

    with open(
        "doc_ids.pkl",
        "rb"
    ) as f:

        semantic_doc_ids = pickle.load(f)

    return index, semantic_doc_ids


index, semantic_doc_ids = load_faiss()

# =====================================================
# RRF
# =====================================================

def reciprocal_rank_fusion(
    rankings,
    k=60
):

    scores = {}

    for ranking in rankings:

        for rank, doc_id in enumerate(
            ranking
        ):

            if doc_id not in scores:

                scores[doc_id] = 0

            scores[doc_id] += (
                1 / (k + rank + 1)
            )

    return sorted(

        scores.items(),

        key=lambda x: x[1],

        reverse=True
    )


# =====================================================
# UI
# =====================================================

st.title(
    "📚 Sistem Temu Kembali Informasi Dokumen Hukum"
)

st.markdown(
    """
Menggunakan:

- BM25
- Semantic Search (SBERT + FAISS)
- Hybrid Search (RRF)
- Cross Encoder Re-Ranking
"""
)

query = st.text_input(
    "Masukkan Query"
)

# =====================================================
# SEARCH
# =====================================================

if st.button("Cari") and query:

    # ==========================================
    # BM25
    # ==========================================

    token_query = preprocessing_teks(
        query
    )

    bm25_scores = bm25.get_scores(
        token_query
    )

    bm25_top_idx = np.argsort(
        bm25_scores
    )[::-1][:10]

    bm25_ranking = [

        doc_ids[i]

        for i in bm25_top_idx
    ]

    # ==========================================
    # SEMANTIC
    # ==========================================

    query_embedding = model.encode(
        [query]
    )

    query_embedding = np.array(
        query_embedding
    ).astype("float32")

    distances, indices = index.search(
        query_embedding,
        10
    )

    semantic_ranking = [

        semantic_doc_ids[i]

        for i in indices[0]
    ]

    # ==========================================
    # HYBRID
    # ==========================================

    hybrid_ranking = reciprocal_rank_fusion(
        [
            bm25_ranking,
            semantic_ranking
        ]
    )

    # ==========================================
    # CROSS ENCODER
    # ==========================================

    top_docs = hybrid_ranking[:10]

    pairs = []

    for doc_id, _ in top_docs:

        pairs.append(
            (
                query,
                doc_dict[doc_id]
            )
        )

    ce_scores = cross_encoder.predict(
        pairs
    )

    reranked = []

    for i, (doc_id, _) in enumerate(
        top_docs
    ):

        reranked.append(
            (
                doc_id,
                ce_scores[i]
            )
        )

    reranked.sort(
        key=lambda x: x[1],
        reverse=True
    )

    # ==========================================
    # OUTPUT
    # ==========================================

    col1, col2 = st.columns(2)

    with col1:

        st.subheader(
            "BM25"
        )

        for i, doc in enumerate(
            bm25_ranking,
            start=1
        ):

            st.write(
                f"{i}. {doc}"
            )

        st.subheader(
            "Semantic Search"
        )

        for i, doc in enumerate(
            semantic_ranking,
            start=1
        ):

            st.write(
                f"{i}. {doc}"
            )

    with col2:

        st.subheader(
            "Hybrid (RRF)"
        )

        for i, (doc, score) in enumerate(
            hybrid_ranking[:10],
            start=1
        ):

            st.write(
                f"{i}. {doc} | {score:.4f}"
            )

        st.subheader(
            "Cross Encoder"
        )

        for i, (doc, score) in enumerate(
            reranked,
            start=1
        ):

            st.write(
                f"{i}. {doc} | {score:.4f}"
            )

    st.success(
        "Pencarian selesai."
    )