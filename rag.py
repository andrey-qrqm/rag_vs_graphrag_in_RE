import os
import time
import torch
import numpy as np
import ollama

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sentence_transformers import SentenceTransformer
from huggingface_hub import login, whoami
from pymongo import MongoClient
from pymongo.server_api import ServerApi
from dotenv import load_dotenv
from neo4j_query import OLLAMA_HOST

#-------------------------------- CONFIG --------------------------------

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
login(HF_TOKEN)
TOP_K = 5

MODEL_NAME = "sentence-transformers/multi-qa-mpnet-base-cos-v1"
embed_model = SentenceTransformer(MODEL_NAME)

CHAT_MODEL = os.getenv("OLLAMA_CHATMODEL")
OLLAMA_HOST = os.getenv("OLLAMA_HOST")
_ollama_client = ollama.Client(host=OLLAMA_HOST)

TIME_LOG_PATH = "logs/3b_vsr_singlehop.log"

QUESTIONS_PATH = 'questions/single_hop_blitdraft.txt'

OUTPUT_PATH = "model_answers/llama32/3b/blitdraft/vsr_singlehop.txt"

MONGODB_USER = os.getenv("MONGODB_USER")
MONGODB_PASSWORD = os.getenv("MONGODB_PASSWORD")
MONGO_URI = f"mongodb+srv://{MONGODB_USER}:{MONGODB_PASSWORD}@slmsbt.pto3n3r.mongodb.net/"

client = MongoClient(MONGO_URI, server_api=ServerApi('1'))
collection = client["rag_db"]["pure_blitdraft"]
#-------------------------------- END CONFIG --------------------------------


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


def embed_query(query: str) -> list[float]:
   return embed_model.encode(query).tolist()


def retrieve_mongoDB(query_embedding: list[float]) -> list[dict]:
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

  messages = [
    {"role": "user", "content": enhanced}
  ]

  result = pipe(messages)
  print(result[0]['generated_text'])
  return result[0]['generated_text']


SYSTEM_QUERY_CONTRADICTION =  """You are a contradiction-detection system. 
            Given a requirement statement as an input, respond with "YES Contradiction" if input requirement contradicts any one requirement from the context 
            Otherwise, respond with "NO Contradiction". 
            Contradiction are present in this forms only: 
            Contradictory opposites like e.g. “he is alive” and “he is not alive” are mutually exhaustive and mutually inconsistent. That states that one statement must be true and the other one false, same vice versa. It is mutually impossible for both statements to be true at the same time. 
            Contrary opposites like e.g. “this block is fully red” and “this block is fully black” are also mutually inconsistent, however not exhaustive. They cannot be true at the same time, but they can both be false.
            Subcontraries e.g. “some people can swim” and “some people can’t swim” are mutually consistent. They can be true at the same time, but they cannot be false at the same time. 
            Subalterns e.g. “Some students had good grade” is the subaltern to a “all students had good grade”. If the statement is true, its subaltern is always true, and if the statement is false, its subaltern is correspondingly false
            Dialectic contradictions, often called conflict of goals, are considered incompatible in practice, while not presenting mathematical or logical conflict. The example could be:
            "The system must have highest performance possible”
            "The system must have lowest resource consumption possible”
            Antinomies denote logical structures where truth can be oscillating. Famous example is Plato says, “Socrates speaks the truth”, Socrates says “Plato lies”
            In your answer provide the one requirement from the context that contradicts the input requirement. If there are multiple contradictions, provide only one. And the type of contradiction. If there is no contradiction, respond with "NO Contradiction"."
            """

SYSTEM_QUERY_DUPLICATION = """You are a duplication-detection system.
            Given a requirement statement as an input, respond with "YES Duplication" if input requirement duplicates any one requirement from the context
            Otherwise, respond with "NO Duplication".
            Duplication is defined as two requirements that are semantically equivalent, meaning they convey the same meaning or intent, even if they are worded differently. This can include requirements that use different terminology
            RESPOND JUST WITH THE DUPLICATION OR NO DUPLICATION. If there is duplication, provide the one requirement from the context that duplicates the input requirement. If there are multiple duplications, provide only one."""

SYSTEM_QUERY_QUESTION_ANSWER = """
    ---Role---

    You are a precise question-answering assistant responding to questions about data in the tables provided.

    ---Goal---
    Respond in the shortest possible form that fully and correctly answers the question — a single sentence or phrase whenever one suffices. Do not restate the question, do not add preamble ("Based on the provided data...", "Here's a possible response..."), and do not explain your reasoning unless explicitly asked.

    Extract only the specific fact(s), entities, or values the question asks for. Do not summarize surrounding context, do not add qualifiers, caveats, or elaboration beyond what directly answers the question.

    If you don't know the answer, say so in one short sentence. Do not make anything up.

    Do not include information where the supporting evidence for it is not provided.

    ---Output format---
    Answer directly, with no headers, no bullet points, and no closing summary — unless the question explicitly asks for a list.
    """

def get_output_mongoDB(query: str) -> str:
  context = retrieve_mongoDB(query)
  print(context)
  context = " ".join(c["requirement"] for c in context)

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

  messages = [
    {
        "role": "system",
        "content": (
           SYSTEM_QUERY_CONTRADICTION
           )
    },
    {
        "role": "user",
        "content": f"Input requirement: {query}.\n Context: {context}"
    }
  ]

  result = pipe(messages)
  result_text = result[0]["generated_text"][-1]["content"].strip()
  return result_text


def get_output_mongoDB_ollama(query: str, system_prompt: str) -> str:
    t0 = time.perf_counter()
    query_embedding = embed_query(query)
    t1 = time.perf_counter()

    context_docs = retrieve_mongoDB(query_embedding)
    t2 = time.perf_counter()
    context = " ".join(c["requirement"] for c in context_docs)

    resp = _ollama_client.chat(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Input requirement: {query}.\n Context: {context}"},
        ],
        options={"num_predict": 64},
        keep_alive="30m",
    )
    t3 = time.perf_counter()

    embed_time = t1 - t0
    retrieve_time = t2 - t1
    answer_time = t3 - t2

    with open(TIME_LOG_PATH, "a") as f:
        f.write(f"{embed_time};{retrieve_time};{answer_time}\n")

    return resp["message"]["content"].strip()


def contradiction_detection(filename: str) -> None:
    reqs_input = load_chunks(filename)
    for req in reqs_input:
        print(f"\nInput requirement: {req['content']}")
        output = get_output_mongoDB_ollama(req['content'], SYSTEM_QUERY_CONTRADICTION)
        print(f"\nOutput: {output}\n")


def duplication_detection(filename: str) -> None:
    reqs_input = load_chunks(filename)
    for req in reqs_input:
        print(f"\nInput requirement: {req['content']}")
        output = get_output_mongoDB_ollama(req['content'], SYSTEM_QUERY_DUPLICATION)
        print(f"\nOutput: {output}\n")


def question_answer(filename: str) -> None:
    reqs_input = load_chunks(filename)
    output_path = OUTPUT_PATH
    for req in reqs_input:
        print(f"\nInput requirement: {req['content']}")
        output = get_output_mongoDB_ollama(req['content'], SYSTEM_QUERY_QUESTION_ANSWER)
        with open(output_path, "a") as f:
            f.write(output + "<END>\n")
        print(f"\nOutput: {output}\n")


#output = get_output_mongoDB(query)
#print("OUTPUT MONGO_DB: ", output)

#reqs_input_fname = "contradictions/contradictions_blitdraft.txt"
#contradiction_detection(reqs_input_fname)
#duplication_detection(reqs_input_fname)

qa_input_fname = QUESTIONS_PATH
question_answer(qa_input_fname)

OLLAMA_HOST = os.getenv("OLLAMA_HOST")
OLLAMA_CHATMODEL = os.getenv("OLLAMA_CHATMODEL")
_ollama_client = ollama.Client(host=OLLAMA_HOST)
