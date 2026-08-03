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

def load_chunks(filename: str) -> list[dict]:
    with open(filename, "r", encoding="utf-8") as f:
        return [{"content": line.rstrip("\n"), "source": filename} for line in f]

FILENAME = "pure_dataset/XMLZIPFile/2010-blitdraft.xml"
#chunks = load_chunks(FILENAME)

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
    chunks = load_chunks(FILENAME)

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
    {
        "role": "system",
        "content": (
            "You are a contradiction-detection system. "
            "Given a requirement statement as an input, respond with Contradiction if input requirement contradicts any one requirement from the context "
            "Otherwise, respond with No Contradiction. "
            "Contradiction are present in this forms only: "
            """Contradictory opposites like e.g. “he is alive” and “he is not alive” are mutually exhaustive and mutually inconsistent. That states that one statement must be true and the other one false, same vice versa. It is mutually impossible for both statements to be true at the same time. 
            "Contrary opposites like e.g. “this block is fully red” and “this block is fully black” are also mutually inconsistent, however not exhaustive. They cannot be true at the same time, but they can both be false.
            Subcontraries e.g. “some people can swim” and “some people can’t swim” are mutually consistent. They can be true at the same time, but they cannot be false at the same time. 
            Subalterns e.g. “Some students had good grade” is the subaltern to a “all students had good grade”. If the statement is true, its subaltern is always true, and if the statement is false, its subaltern is correspondingly false
            Dialectic contradictions, often called conflict of goals, are considered incompatible in practice, while not presenting mathematical or logical conflict. The example could be:
            “The system must have highest performance possible”
            “The system must have lowest resource consumption possible”
            Antinomies denote logical structures where truth can be oscillating. Famous example is Plato says, “Socrates speaks the truth”, Socrates says “Plato lies”
            """
            "In your answer provide the one requirement from the context that contradicts the input requirement. If there are multiple contradictions, provide only one. And the type of contradiction. If there is no contradiction, respond with No Contradiction."
        )
    },
    {
        "role": "user",
        "content": f"Input requirement: {query}.\n Context: {context}"
    }
  ]

  result = pipe(messages)
  result_text = result[0]["generated_text"][-1]["content"].strip()
  #print(result_text)
  return result_text

query = """Does this requirement has any contradictions with any other requirement? If yes - show the requirement it contradicts.:
 Query - The system shall not display operational page to any user besides with administrator role."""
#print("CONTEXT MONGO_DB: ", retrieve_mongoDB(query))

def contradiction_detection(filename: str) -> None:
    reqs_input = load_chunks(filename)
    for req in reqs_input:
        print(f"\nInput requirement: {req['content']}")
        output = get_output_mongoDB(req['content'])
        print(f"\nOutput: {output}\n")


#output = get_output_mongoDB(query)
#print("OUTPUT MONGO_DB: ", output)
reqs_input_fname = "contradictions_blitdraft.txt"
contradiction_detection(reqs_input_fname)
