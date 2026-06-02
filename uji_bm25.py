import os
import re
import pandas as pd
import fitz  # PyMuPDF

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from rank_bm25 import BM25Okapi

# =====================================================
# 1. SETUP SASTRAWI
# =====================================================

stemmer = StemmerFactory().create_stemmer()
stopword = StopWordRemoverFactory().create_stop_word_remover()

def preprocessing_teks(teks):
    teks = re.sub(r'[^a-zA-Z\s]', ' ', teks.lower())
    teks = stopword.remove(teks)
    return stemmer.stem(teks).split()

# =====================================================
# 2. EKSTRAKSI PDF
# =====================================================

def ekstrak_semua_pdf(folders):

    dokumen_korpus = []
    doc_ids = []

    for folder in folders:

        for file_name in os.listdir(folder):

            if file_name.endswith('.pdf'):

                # =================================================
                # AMBIL NAMA FILE TANPA .pdf
                # Contoh:
                # UMKM_03_UU_33.pdf
                # menjadi:
                # UMKM_03_UU_33
                # =================================================

                doc_id = os.path.splitext(file_name)[0]

                doc_ids.append(doc_id)

                teks = ""

                path_pdf = os.path.join(folder, file_name)

                try:

                    doc = fitz.open(path_pdf)

                    for page in doc:
                        teks += page.get_text() + " "

                    hasil_preprocessing = preprocessing_teks(teks)

                    dokumen_korpus.append(hasil_preprocessing)

                    print(f"Berhasil memproses: {file_name}")

                except Exception as e:
                    print(f"Gagal memproses {file_name}: {e}")

    return doc_ids, dokumen_korpus

# =====================================================
# 3. BUILD BM25
# =====================================================

print("Membangun Indeks BM25...")

folders_pdf = [
    "TOPIK_1_UMKM",
    "TOPIK_2_KETENAGAKERJAAN"
]

doc_ids, tokenized_corpus = ekstrak_semua_pdf(folders_pdf)

bm25 = BM25Okapi(tokenized_corpus)

print(f"\nTotal dokumen berhasil dimuat: {len(doc_ids)}")

# =====================================================
# 4. LOAD GROUND TRUTH
# =====================================================

df_gt = pd.read_excel("Ground_Truth.xlsx")

precision_list = []
recall_list = []

print("\nMenjalankan evaluasi...")

# =====================================================
# 5. EVALUASI
# =====================================================

for index, row in df_gt.iterrows():

    query = str(row['Teks_Query'])

    # ============================================
    # Ground Truth
    # Format Excel:
    # UMKM_09_PP_05,UMKM_11_PP_24
    # ============================================

    kunci_jawaban = [
        x.strip()
        for x in str(row['Doc_ID_Relevan']).split(',')
    ]

    # ============================================
    # PREPROCESS QUERY
    # ============================================

    token_query = preprocessing_teks(query)

    # ============================================
    # HITUNG BM25 SCORE
    # ============================================

    skor = bm25.get_scores(token_query)

    # ============================================
    # TOP 5 DOKUMEN
    # ============================================

    top_5_idx = sorted(
        range(len(skor)),
        key=lambda i: skor[i],
        reverse=True
    )[:5]

    hasil_retrieval = [
        doc_ids[i]
        for i in top_5_idx
    ]

    # ============================================
    # HITUNG HITS
    # ============================================

    hits = len(
        set(hasil_retrieval)
        &
        set(kunci_jawaban)
    )

    precision = hits / 5

    recall = (
        hits / len(kunci_jawaban)
        if len(kunci_jawaban) > 0
        else 0
    )

    precision_list.append(precision)
    recall_list.append(recall)

    # ============================================
    # DEBUG OUTPUT
    # ============================================

    print("\n====================================")
    print(f"QUERY           : {query}")
    print(f"HASIL RETRIEVAL : {hasil_retrieval}")
    print(f"GROUND TRUTH    : {kunci_jawaban}")
    print(f"Hits            : {hits}")
    print(f"Precision@5     : {precision:.2f}")
    print(f"Recall@5        : {recall:.2f}")

# =====================================================
# 6. HASIL AKHIR
# =====================================================

avg_precision = sum(precision_list) / len(precision_list)
avg_recall = sum(recall_list) / len(recall_list)

print("\n====================================")
print("--- HASIL AKHIR ---")
print(f"Rata-rata Precision@5: {avg_precision:.2f}")
print(f"Rata-rata Recall@5: {avg_recall:.2f}")