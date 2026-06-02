import os
import re
import fitz
import faiss
import pickle
import pandas as pd
import numpy as np

from rank_bm25 import BM25Okapi

from sentence_transformers import (
    SentenceTransformer,
    CrossEncoder
)

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# =====================================================
# PREPROCESSING
# =====================================================

stemmer = StemmerFactory().create_stemmer()

stopword = StopWordRemoverFactory().create_stop_word_remover()

def preprocessing_teks(teks):

    teks = teks.lower()

    teks = re.sub(r'[^a-zA-Z\s]', ' ', teks)

    teks = stopword.remove(teks)

    return stemmer.stem(teks).split()

# =====================================================
# LOAD PDF
# =====================================================

def load_documents(folder_list):

    documents = []

    doc_ids = []

    for folder in folder_list:

        for file_name in os.listdir(folder):

            if file_name.endswith(".pdf"):

                path_pdf = os.path.join(folder, file_name)

                teks = ""

                try:

                    doc = fitz.open(path_pdf)

                    for page in doc:

                        teks += page.get_text() + " "

                    documents.append(teks)

                    doc_ids.append(
                        file_name.replace(".pdf", "")
                    )

                except Exception as e:

                    print(
                        f"Gagal membaca {file_name}: {e}"
                    )

    return doc_ids, documents

# =====================================================
# LOAD DATASET
# =====================================================

folders = [
    "TOPIK_1_UMKM",
    "TOPIK_2_KETENAGAKERJAAN"
]

print("Loading Documents...")

doc_ids, documents = load_documents(folders)

doc_dict = {

    doc_id: doc

    for doc_id, doc in zip(doc_ids, documents)
}

print(f"Total Dokumen: {len(doc_ids)}")

# =====================================================
# BUILD BM25
# =====================================================

print("Building BM25...")

tokenized_corpus = [

    preprocessing_teks(doc)

    for doc in documents
]

bm25 = BM25Okapi(tokenized_corpus)

# =====================================================
# LOAD SEMANTIC MODEL
# =====================================================

print("Loading SBERT...")

model = SentenceTransformer(
    "paraphrase-multilingual-MiniLM-L12-v2"
)

# =====================================================
# LOAD CROSS ENCODER
# =====================================================

print("Loading Cross Encoder...")

cross_encoder = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

# =====================================================
# LOAD FAISS
# =====================================================

print("Loading FAISS...")

index = faiss.read_index(
    "index_semantic.faiss"
)

with open("doc_ids.pkl", "rb") as f:

    semantic_doc_ids = pickle.load(f)

# =====================================================
# RRF
# =====================================================

def reciprocal_rank_fusion(
    rankings,
    k=60
):

    scores = {}

    for ranking in rankings:

        for rank, doc_id in enumerate(ranking):

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
# EVALUASI
# =====================================================

df_gt = pd.read_excel(
    "Ground_Truth.xlsx"
)

hasil = {

    "BM25": {
        "precision": [],
        "recall": []
    },

    "Semantic": {
        "precision": [],
        "recall": []
    },

    "Hybrid": {
        "precision": [],
        "recall": []
    },

    "CrossEncoder": {
        "precision": [],
        "recall": []
    }
}

# =====================================================
# LOOP QUERY
# =====================================================

for _, row in df_gt.iterrows():

    query = str(
        row["Teks_Query"]
    )

    ground_truth = [

        x.strip()

        for x in str(
            row["Doc_ID_Relevan"]
        ).split(",")
    ]

    # =================================================
    # BM25
    # =================================================

    token_query = preprocessing_teks(
        query
    )

    bm25_scores = bm25.get_scores(
        token_query
    )

    bm25_idx = np.argsort(
        bm25_scores
    )[::-1][:5]

    bm25_result = [

        doc_ids[i]

        for i in bm25_idx
    ]

    # =================================================
    # SEMANTIC
    # =================================================

    query_embedding = model.encode(
        [query]
    )

    query_embedding = np.array(
        query_embedding
    ).astype("float32")

    distances, indices = index.search(
        query_embedding,
        5
    )

    semantic_result = [

        semantic_doc_ids[i]

        for i in indices[0]
    ]

    # =================================================
    # HYBRID
    # =================================================

    bm25_top10 = [

        doc_ids[i]

        for i in np.argsort(
            bm25_scores
        )[::-1][:10]
    ]

    distances, indices = index.search(
        query_embedding,
        10
    )

    semantic_top10 = [

        semantic_doc_ids[i]

        for i in indices[0]
    ]

    hybrid_result = [

        doc

        for doc, _ in reciprocal_rank_fusion(
            [
                bm25_top10,
                semantic_top10
            ]
        )[:5]
    ]

    # =================================================
    # CROSS ENCODER
    # =================================================

    top_docs = [

        doc

        for doc, _ in reciprocal_rank_fusion(
            [
                bm25_top10,
                semantic_top10
            ]
        )[:10]
    ]

    pairs = [

        (
            query,
            doc_dict[d]
        )

        for d in top_docs
    ]

    ce_scores = cross_encoder.predict(
        pairs
    )

    reranked = list(
        zip(
            top_docs,
            ce_scores
        )
    )

    reranked.sort(
        key=lambda x: x[1],
        reverse=True
    )

    cross_result = [

        doc

        for doc, _ in reranked[:5]
    ]

    # =================================================
    # HITUNG METRIK
    # =================================================

    metode = {

        "BM25": bm25_result,

        "Semantic": semantic_result,

        "Hybrid": hybrid_result,

        "CrossEncoder": cross_result
    }

    for nama, retrieval in metode.items():

        hits = len(
            set(retrieval)
            &
            set(ground_truth)
        )

        precision = hits / 5

        recall = (
            hits / len(ground_truth)
        )

        hasil[nama][
            "precision"
        ].append(
            precision
        )

        hasil[nama][
            "recall"
        ].append(
            recall
        )

# =====================================================
# HASIL AKHIR
# =====================================================

print("\n")

print("=" * 60)

print("HASIL EVALUASI")

print("=" * 60)

for metode in hasil:

    avg_precision = np.mean(
        hasil[metode]["precision"]
    )

    avg_recall = np.mean(
        hasil[metode]["recall"]
    )

    print(
        f"{metode:15s}"
        f" Precision@5 = {avg_precision:.4f}"
        f" | Recall@5 = {avg_recall:.4f}"
    )