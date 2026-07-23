import os
from pymongo import MongoClient
from pymongo.server_api import ServerApi
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()
MONGODB_USER = os.getenv("MONGODB_USER")
MONGODB_PASSWORD = os.getenv("MONGODB_PASSWORD")

MONGO_URI = f"mongodb+srv://{MONGODB_USER}:{MONGODB_PASSWORD}@slmsbt.pto3n3r.mongodb.net/"

MODEL_NAME = "sentence-transformers/multi-qa-mpnet-base-cos-v1"
FILENAME = "software_requirements_data/nfr.txt"

client = MongoClient(MONGO_URI, server_api=ServerApi('1'))
collection = client["rag_db"]["nfr_chunks"]

def load_chunks() -> list[dict]:
    with open(FILENAME, "r", encoding="utf-8") as f:
        return [{"content": line.rstrip("\n"), "source": FILENAME} for line in f]

def ingest():
    model = SentenceTransformer(MODEL_NAME)
    chunks = load_chunks()
    texts = [c["content"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=True)

    docs = [
        {"content": c["content"], "source": c["source"], "embedding": emb.tolist()}
        for c, emb in zip(chunks, embeddings)
    ]

    collection.delete_many({})  # optional: clear old data before re-ingesting
    collection.insert_many(docs)
    print(f"Inserted {len(docs)} chunks")

if __name__ == "__main__":
    ingest()