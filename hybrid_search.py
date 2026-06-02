import os
import re
import faiss
import pickle
import fitz
import numpy as np

from rank_bm25 import BM25Okapi

from sentence_transformers import (
    SentenceTransformer,
    CrossEncoder
)

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# =====================================================
# SETUP PREPROCESSING
# =====================================================

stemmer = StemmerFactory().create_stemmer()

stopword = StopWordRemoverFactory().create_stop_word_remover()

def preprocessing_teks(teks):

    teks = teks.lower()

    teks = re.sub(r'[^a-zA-Z\s]', ' ', teks)

    tokens = teks.split()

    hasil = []

    for token in tokens:

        if len(token) <= 2:
            continue

        token = stemmer.stem(token)

        hasil.append(token)

    return hasil

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

                    print(f"Loaded: {file_name}")

                except Exception as e:

                    print(f"Gagal: {file_name} | {e}")

    return doc_ids, documents

# =====================================================
# LOAD DATA
# =====================================================

folders = [
    "TOPIK_1_UMKM",
    "TOPIK_2_KETENAGAKERJAAN"
]

doc_ids, documents = load_documents(folders)

# =====================================================
# DOCUMENT MAPPING
# =====================================================

doc_dict = {

    doc_id: doc

    for doc_id, doc in zip(doc_ids, documents)
}

# =====================================================
# BM25 INDEX
# =====================================================

print("\nMembangun BM25...")

tokenized_corpus = [

    preprocessing_teks(doc)

    for doc in documents
]

bm25 = BM25Okapi(tokenized_corpus)

# =====================================================
# LOAD SEMANTIC MODEL
# =====================================================

print("\nLoading Semantic Model...")

model = SentenceTransformer(
    'paraphrase-multilingual-MiniLM-L12-v2'
)

# =====================================================
# LOAD CROSS ENCODER
# =====================================================

print("\nLoading Cross Encoder...")

cross_encoder = CrossEncoder(
    'cross-encoder/ms-marco-MiniLM-L-6-v2'
)

# =====================================================
# LOAD FAISS INDEX
# =====================================================

print("\nLoading FAISS Index...")

index = faiss.read_index("index_semantic.faiss")

with open("doc_ids.pkl", "rb") as f:

    semantic_doc_ids = pickle.load(f)

# =====================================================
# RRF FUNCTION
# =====================================================

def reciprocal_rank_fusion(rankings, k=60):

    scores = {}

    for ranking in rankings:

        for rank, doc_id in enumerate(ranking):

            if doc_id not in scores:

                scores[doc_id] = 0

            scores[doc_id] += 1 / (k + rank + 1)

    return sorted(

        scores.items(),

        key=lambda x: x[1],

        reverse=True
    )

# =====================================================
# QUERY LOOP
# =====================================================

while True:

    print("\n===================================")

    query = input(
        "Masukkan query (ketik 'exit' untuk keluar): "
    )

    if query.lower() == "exit":

        break

    # =========================================
    # BM25 SEARCH
    # =========================================

    token_query = preprocessing_teks(query)

    bm25_scores = bm25.get_scores(token_query)

    bm25_top_idx = np.argsort(
        bm25_scores
    )[::-1][:10]

    bm25_ranking = [

        doc_ids[i]

        for i in bm25_top_idx
    ]

    # =========================================
    # SEMANTIC SEARCH
    # =========================================

    query_embedding = model.encode([query])

    query_embedding = np.array(
        query_embedding
    ).astype('float32')

    distances, indices = index.search(
        query_embedding,
        10
    )

    semantic_ranking = [

        semantic_doc_ids[i]

        for i in indices[0]
    ]

    # =========================================
    # HYBRID RRF
    # =========================================

    final_ranking = reciprocal_rank_fusion(
        [bm25_ranking, semantic_ranking]
    )

    # =========================================
    # CROSS ENCODER RE-RANKING
    # =========================================

    top_docs = final_ranking[:10]

    pairs = []

    for doc_id, _ in top_docs:

        doc_text = doc_dict[doc_id]

        pairs.append(
            (query, doc_text)
        )

    ce_scores = cross_encoder.predict(pairs)

    reranked = []

    for i, (doc_id, _) in enumerate(top_docs):

        reranked.append(
            (doc_id, ce_scores[i])
        )

    reranked.sort(
        key=lambda x: x[1],
        reverse=True
    )

    # =========================================
    # OUTPUT BM25
    # =========================================

    print("\n===== HASIL BM25 =====")

    for i, doc in enumerate(
        bm25_ranking,
        start=1
    ):

        print(f"{i}. {doc}")

    # =========================================
    # OUTPUT SEMANTIC
    # =========================================

    print("\n===== HASIL SEMANTIC =====")

    for i, doc in enumerate(
        semantic_ranking,
        start=1
    ):

        print(f"{i}. {doc}")

    # =========================================
    # OUTPUT HYBRID RRF
    # =========================================

    print("\n===== HASIL HYBRID (RRF) =====")

    for i, (doc, score) in enumerate(
        final_ranking[:10],
        start=1
    ):

        print(
            f"{i}. {doc} | Score: {score:.4f}"
        )

    # =========================================
    # OUTPUT CROSS ENCODER
    # =========================================

    print("\n===== HASIL CROSS ENCODER =====")

    for i, (doc, score) in enumerate(
        reranked,
        start=1
    ):

        print(
            f"{i}. {doc} | CE Score: {score:.4f}"
        )