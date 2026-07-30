import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sentence_transformers import SentenceTransformer
from huggingface_hub import login, whoami
from pymongo import MongoClient
from pymongo.server_api import ServerApi
from dotenv import load_dotenv

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
login(HF_TOKEN)
TOP_K = 5

MODEL_NAME = "sentence-transformers/multi-qa-mpnet-base-cos-v1"
MONGODB_USER = os.getenv("MONGODB_USER")
MONGODB_PASSWORD = os.getenv("MONGODB_PASSWORD")
MONGO_URI = f"mongodb+srv://{MONGODB_USER}:{MONGODB_PASSWORD}@slmsbt.pto3n3r.mongodb.net/"
client = MongoClient(MONGO_URI, server_api=ServerApi('1'))
collection = client["rag_db"]["pure_blitdraft"]

def load_chunks() -> list[dict]:
    with open(FILENAME, "r", encoding="utf-8") as f:
        return [{"content": line.rstrip("\n"), "source": FILENAME} for line in f]
    
FILENAME = "pure_dataset/XMLZIPFile/2010-blitdraft.xml"
#chunks = load_chunks()

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def embed_texts(model: SentenceTransformer, texts: list[str]) -> np.ndarray:
    return model.encode(texts, show_progress_bar=False)

def retrieve(query: str) -> list[dict]:
    """
    Return top K most relevant passages for the query.

    Each result must have:
      - "content": str       — the passage content
      - "source": str     — source filename or "glossary"
      - "score": float    — relevance score (higher = better)
    """
    model = SentenceTransformer(MODEL_NAME)
    chunks = load_chunks()

    if not chunks:
        return []

    texts = [c["content"] for c in chunks]
    chunk_embeddings = embed_texts(model, texts)
    query_embedding = model.encode(query)

    scored = [
        {**chunk, "score": cosine_similarity(query_embedding, emb)}
        for chunk, emb in zip(chunks, chunk_embeddings)
    ]

    return sorted(scored, key=lambda x: x["score"], reverse=True)[:TOP_K]


def retrieve_mongoDB(query: str) -> list[dict]:
  model = SentenceTransformer(MODEL_NAME)
  query_embedding = model.encode(query).tolist()

  pipeline = [
      {
          "$vectorSearch": {
              "index": "pure_blitdraft_index",
              "path": "embedding",
              "queryVector": query_embedding,
              "numCandidates": 100,
              "limit": TOP_K
          }
      },
      {
          "$project": {
              "_id": 0,
              "id": 1,
              "branch": 1,
              "module": 1,
              "usecase": 1,
              "action": 1,
              "requirement": 1,
              "score": {"$meta": "vectorSearchScore"}
          }
      }
  ]

  return list(collection.aggregate(pipeline))

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
pipe = pipeline(
    'text-generation',
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.float16,
    device_map="auto",
    max_new_tokens=128,
)

def get_context(query: str) -> list[dict]:
  context = retrieve(query)
  return context


def get_output(query: str) -> str:
  context = get_context(query)
  context = " ".join(c["requirement"] for c in context)

  enhanced = f"""
    QUERY - {query},
    CONTEXT - {context}
  """

  pipe = pipeline(
    'text-generation',
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.float16,
    device_map="auto",
    max_new_tokens=128,
  )

  messages = [
    {"role": "user", "content": enhanced}
  ]

  result = pipe(messages)
  print(result[0]['generated_text'])
  return result[0]['generated_text']


def get_output_mongoDB(query: str) -> str:
  context = retrieve_mongoDB(query)
  print(context)
  context = " ".join(c["requirement"] for c in context)

  enhanced = f"""
    QUERY - {query},
    CONTEXT - {context}
  """

  pipe = pipeline(
    'text-generation',
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.float16,
    device_map="auto",
    max_new_tokens=128,
  )

  messages = [
    {"role": "user", "content": enhanced}
  ]

  result = pipe(messages)
  print(result[0]['generated_text'])
  return result[0]['generated_text']

query = """Does this requirement has any contradictions with any other requirement? If yes - show the requirement it contradicts.:
 Query - The system shall not display operational page to any user besides with administrator role."""
print("CONTEXT MONGO_DB: ", retrieve_mongoDB(query))

output = get_output_mongoDB(query)
print("OUTPUT MONGO_DB: ", output)
