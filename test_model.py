from sentence_transformers import SentenceTransformer

print("loading model")

model = SentenceTransformer(
    "paraphrase-multilingual-MiniLM-L12-v2"
)

print("berhasil")