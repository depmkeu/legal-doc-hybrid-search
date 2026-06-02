import os
import fitz
import faiss
import pickle
import numpy as np

from sentence_transformers import SentenceTransformer

# ============================================
# LOAD MODEL INDOSBERT
# ============================================

print("Loading model embedding...")

model = SentenceTransformer(
    'paraphrase-multilingual-MiniLM-L12-v2'
)

# ============================================
# EKSTRAKSI PDF
# ============================================

def ekstrak_teks_pdf(folder_list):

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

                    # simpan doc id
                    doc_id = file_name.replace(".pdf", "")

                    doc_ids.append(doc_id)

                    documents.append(teks)

                    print(f"Berhasil: {file_name}")

                except Exception as e:
                    print(f"Gagal membaca {file_name}: {e}")

    return doc_ids, documents

# ============================================
# LOAD DATA PDF
# ============================================

folders = [
    "TOPIK_1_UMKM",
    "TOPIK_2_KETENAGAKERJAAN"
]

doc_ids, documents = ekstrak_teks_pdf(folders)

print(f"\nTotal dokumen: {len(documents)}")

# ============================================
# BUAT EMBEDDING
# ============================================

print("\nMembuat embedding vector...")

embeddings = model.encode(
    documents,
    show_progress_bar=True
)

embeddings = np.array(embeddings).astype('float32')

print("Ukuran embedding:", embeddings.shape)

# ============================================
# BUILD FAISS INDEX
# ============================================

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(embeddings)

print("\nFAISS index berhasil dibuat!")

# ============================================
# SIMPAN INDEX
# ============================================

faiss.write_index(index, "index_semantic.faiss")

with open("doc_ids.pkl", "wb") as f:
    pickle.dump(doc_ids, f)

print("\nSemua berhasil disimpan!")