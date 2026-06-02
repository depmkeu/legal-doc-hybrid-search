# 📚 Sistem Temu Kembali Informasi (STKI) - Dokumen Hukum

Aplikasi pencarian dokumen hukum hibrida yang dibangun menggunakan pendekatan Lexical (BM25) dan Semantic Search (IndoSBERT), serta disempurnakan dengan teknik Reciprocal Rank Fusion (RRF) dan Re-ranking (Cross-Encoder).

## 🚀 Tautan Aplikasi Web
**Aplikasi dapat diakses melalui:** [🔗 https://legal-doc-hybrid-search.streamlit.app/]

## 🛠️ Teknologi yang Digunakan
*   **Antarmuka Web:** Streamlit
*   **Pencarian Leksikal:** BM25 (rank-bm25)
*   **Pencarian Semantik:** Sentence-Transformers (IndoSBERT: `paraphrase-multilingual-MiniLM-L12-v2`)
*   **Penyimpanan Vektor:** FAISS (Facebook AI Similarity Search)
*   **Re-ranking AI:** Cross-Encoder (`ms-marco-MiniLM-L-6-v2`)
*   **Pemrosesan Teks:** PyMuPDF (fitz), Sastrawi (Stemmer & Stopword)

## 📂 Alur Kerja Sistem
1.  **Ekstraksi Data:** Mengekstrak 40 dokumen PDF (UU & PP Ketenagakerjaan dan UMKM) menjadi teks.
2.  **Indexing:** Membangun indeks BM25 dan indeks vektor FAISS.
3.  **Hybrid Retrieval (RRF):** Menggabungkan Top-10 hasil dari BM25 dan IndoSBERT menggunakan metode *Reciprocal Rank Fusion*.
4.  **Re-ranking:** Melakukan evaluasi ulang terhadap hasil penggabungan menggunakan model *Cross-Encoder* untuk mendapatkan skor relevansi paling presisi pada Top-5.
