from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sentence-transformers/LaBSE")
print("LaBSE loaded:", model)
