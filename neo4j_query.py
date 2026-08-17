"""
graphrag_query.py

Local GraphRAG-style query pipeline:
  1. Embed the user query with a local Ollama embedding model.
  2. Vector-search Neo4j for seed __Entity__ nodes.
  3. Pivot on each seed's parent __Chunk__ and pull the full set of
     Subject/Action/Object/Condition entities attached to that chunk,
     reconstructing the complete requirement tuple (not just whichever
     single component the vector search happened to match).
  4. Pull relationships between the reconstructed components, plus the
     community summaries and findings those components belong to.
  5. Assemble a context block and send it + the query to a local Ollama
     chat model to produce the final answer.

Schema (confirmed via db.schema.visualization()):
  Nodes:
    (:__Entity__ {id, name, description, embedding})
      -- also multi-labeled with one of :Subject / :Action / :Object / :Condition
    (:__Chunk__  {id, text, embedding})
    (:__Document__ {id})
    (:__Community__ {community, title, summary})
    (:Finding {summary})
  Relationships:
    (:__Entity__)-[:RELATED {description}]->(:__Entity__)
    (:__Chunk__)-[:HAS_ENTITY]->(:__Entity__)
    (:__Chunk__)-[:PART_OF]->(:__Document__)
    (:__Entity__)-[:IN_COMMUNITY]->(:__Community__)
    (:__Community__)-[:HAS_FINDING]->(:Finding)

If your import used different labels/properties, adjust the constants
and Cypher in `retrieve_context()` accordingly.

Requires:
    pip install neo4j ollama
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field

import ollama
from neo4j import GraphDatabase, Driver
from openai import OpenAI

import time

import os
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

load_dotenv()

NEO4J_URI=os.getenv("NEO4J_URL")
NEO4J_USERNAME=os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD=os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE=os.getenv("NEO4J_DATABASE")


OLLAMA_HOST = os.getenv("OLLAMA_HOST")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

CHAT_MODEL = os.getenv("OLLAMA_CHATMODEL")         
# Name of the vector index created on __Entity__.embedding
# e.g. CREATE VECTOR INDEX entity_embedding IF NOT EXISTS
#      FOR (e:__Entity__) ON (e.embedding)
#      OPTIONS {indexConfig: {`vector.dimensions`: 768, `vector.similarity_function`: 'cosine'}}
ENTITY_VECTOR_INDEX = "entity_embedding"

TOP_K_ENTITIES = 5
MAX_RELATIONS_PER_ENTITY = 5

OUTPUT_PATH = "model_answers/llama32/3b/blitdraft/neo4j_singlehop.txt"
LOG_PATH = "logs/neo4j_graphrag_timelog_3b_singlehop.log"
INPUT_PATH = "questions/single_hop_blitdraft.txt"

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class GraphContext:
    requirements: list[dict] = field(default_factory=list)   # reconstructed S/A/O/C tuples
    relationships: list[dict] = field(default_factory=list)
    communities: list[dict] = field(default_factory=list)
    findings: list[dict] = field(default_factory=list)

    def to_prompt_block(self) -> str:
        parts = []

        if self.requirements:
            req_lines = []
            for r in self.requirements:
                comp_str = "; ".join(
                    f"{role}: {name}" for role, name in r["components"]
                )
                req_lines.append(f"- Source text: {r['chunk_text'].strip()}\n  Components -> {comp_str}")
            parts.append("## Retrieved requirement components (grouped by source chunk)\n" + "\n".join(req_lines))

        if self.relationships:
            rel_lines = "\n".join(
                f"- {r['source']} ({r['source_label']}) -> {r['target']} ({r['target_label']}): "
                f"{(r.get('description') or '').strip()}"
                for r in self.relationships
            )
            parts.append(f"## Relationships between components\n{rel_lines}")

        if self.communities:
            comm_lines = "\n\n".join(
                f"[{c.get('title', c['id'])}]\n{(c.get('summary') or '').strip()}"
                for c in self.communities
            )
            parts.append(f"## Community summaries\n{comm_lines}")

        if self.findings:
            finding_lines = "\n".join(f"- {f.get('summary', '').strip()}" for f in self.findings)
            parts.append(f"## Community findings\n{finding_lines}")

        return "\n\n".join(parts) if parts else "(no context retrieved)"

    def print(self) -> None:
        print(self.to_prompt_block())
        print("Retrieved requirements:", len(self.requirements))
        print("Retrieved relationships:", len(self.relationships))
        print("Retrieved communities:", len(self.communities))
        print("Retrieved findings:", len(self.findings))

# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------
EMBEDDING_MODEL = "text-embedding-3-small"  # must match graphrag's embedding_models config
EMBEDDING_DIMENSIONS = 1536  # default output dim for text-embedding-3-small; confirm
                              # against your vector index config if you set a custom
                              # `dimensions` truncation value in the GraphRAG settings
 
_openai_client = OpenAI() # reads OPENAI_API_KEY from env
 
 
def embed_query(query: str, model: str = EMBEDDING_MODEL) -> list[float]:
    resp = _openai_client.embeddings.create(model=model, input=query)
    return resp.data[0].embedding
 
 

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def retrieve_context(driver: Driver, query_embedding: list[float]) -> GraphContext:

    ctx = GraphContext()

    with driver.session() as session:
        # 1. Seed entities via vector similarity search (over all __Entity__ nodes,
        #    regardless of which of Subject/Action/Object/Condition they also carry)
        seed_rows = session.run(
             f"""
                MATCH (e:__Entity__)
                SEARCH e IN (
                    VECTOR INDEX `{ENTITY_VECTOR_INDEX}`
                    FOR $embedding
                    LIMIT $k
                ) SCORE AS score
                RETURN e.id AS id, e.name AS name, score
                ORDER BY score DESC
                """,
            k=TOP_K_ENTITIES,
            embedding=query_embedding,
        ).data()

        if not seed_rows:
            return ctx

        seed_ids = [r["id"] for r in seed_rows]

        # 2. For each seed entity's parent chunk(s), pull the full set of
        #    Subject/Action/Object/Condition entities attached to that chunk,
        #    reconstructing the complete requirement.
        req_rows = session.run(
            """
            MATCH (e:__Entity__)<-[:HAS_ENTITY]-(chunk:__Chunk__)
            WHERE e.id IN $ids
            MATCH (chunk)-[:HAS_ENTITY]->(comp:__Entity__)
            WHERE comp:Subject OR comp:Action OR comp:Object OR comp:Condition
            WITH chunk, comp
            ORDER BY chunk.id
            RETURN chunk.id AS chunk_id, chunk.text AS chunk_text,
                   collect(DISTINCT {
                       role: CASE
                           WHEN comp:Subject THEN 'Subject'
                           WHEN comp:Action THEN 'Action'
                           WHEN comp:Object THEN 'Object'
                           WHEN comp:Condition THEN 'Condition'
                           ELSE 'Other'
                       END,
                       name: comp.name,
                       id: comp.id
                   }) AS components
            """,
            ids=seed_ids,
        ).data()

        ctx.requirements = [
            {
                "chunk_id": r["chunk_id"],
                "chunk_text": r["chunk_text"],
                "components": [(c["role"], c["name"]) for c in r["components"]],
            }
            for r in req_rows
        ]

        component_ids = {
            c["id"] for r in req_rows for c in r["components"]
        } | set(seed_ids)

        # 3. Relationships between all reconstructed components (and the seeds),
        #    so cross-requirement links (e.g. shared Subject) are still visible.
        rel_rows = session.run(
            """
            MATCH (a:__Entity__)-[r:RELATED]-(b:__Entity__)
            WHERE a.id IN $ids AND b.id IN $ids AND a.id <> b.id
            RETURN DISTINCT a.name AS source, labels(a) AS source_labels,
                   b.name AS target, labels(b) AS target_labels,
                   r.description AS description
            LIMIT $limit
            """,
            ids=list(component_ids),
            limit=TOP_K_ENTITIES * MAX_RELATIONS_PER_ENTITY,
        ).data()
        ctx.relationships = [
            {
                "source": r["source"],
                "source_label": next((l for l in r["source_labels"] if l != "__Entity__"), ""),
                "target": r["target"],
                "target_label": next((l for l in r["target_labels"] if l != "__Entity__"), ""),
                "description": r["description"],
            }
            for r in rel_rows
        ]

        # 4. Community summaries the components belong to
        comm_rows = session.run(
            """
            MATCH (e:__Entity__)-[:IN_COMMUNITY]->(c:__Community__)
            WHERE e.id IN $ids
            WITH e, c
            ORDER BY c.level DESC
            WITH e, collect(c)[0] AS c
            RETURN DISTINCT c.community AS id, c.title AS title,
                   c.summary AS summary, c.rank AS rank
            """,
            ids=list(component_ids),
        ).data()
        ctx.communities = comm_rows
 
        # 5. Findings attached to those communities (capped per community so a
        #    single large community's findings don't dominate the context)
        if comm_rows:
            community_ids = [c["id"] for c in comm_rows]
            finding_rows = session.run(
                """
                MATCH (c:__Community__)-[:HAS_FINDING]->(f:Finding)
                WHERE c.community IN $cids
                WITH c, f
                ORDER BY c.community
                WITH c, collect(DISTINCT f.summary)[0..3] AS topFindings
                UNWIND topFindings AS summary
                RETURN DISTINCT summary
                """,
                cids=community_ids,
            ).data()
            ctx.findings = finding_rows

    return ctx


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a domain assistant answering questions using a knowledge \
graph retrieved from a document corpus. Use only the provided context. \
If the context does not contain enough information to answer, say so explicitly \
rather than guessing."""


SYSTEM_QUERY_CONTRADICTION =  """You are a contradiction-detection system. 
            Given a requirement statement as an input, respond with "YES Contradiction" if input requirement contradicts any one requirement stored in the system. As a context you receive a knowledge graph retrieved from a document corpus. Use only the provided context. 
            Otherwise, respond with "NO Contradiction". 
            """

SYSTEM_QUERY_QA =  """
    ---Role---

    You are a precise question-answering assistant responding to questions about data in the tables provided.

    ---Goal---
    Respond in the shortest possible form that fully and correctly answers the question — a single sentence or phrase whenever one suffices. Do not restate the question, do not add preamble ("Based on the provided data...", "Here's a possible response..."), and do not explain your reasoning unless explicitly asked.

    Extract only the specific fact(s), entities, or values the question asks for. Do not summarize surrounding context, do not add qualifiers, caveats, or elaboration beyond what directly answers the question.

    If you don't know the answer, say so in one short sentence. Do not make anything up.

    Do not list more than 5 record ids in a single reference. Instead, list the top 5 most relevant record ids and add "+more" to indicate that there are more.

    Do not include information where the supporting evidence for it is not provided.

    ---Output format---
    Answer directly, with no headers, no bullet points, and no closing summary — unless the question explicitly asks for a list.
    """

def generate_answer(query: str, context: GraphContext, model: str, system_prompt: str) -> str:
    client = ollama.Client(host=OLLAMA_HOST)

    user_prompt = f"""Context retrieved from the knowledge graph:

        {context.to_prompt_block()}

        Question: {query}

        Answer using only the context above."""

    resp = client.chat(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        options={"num_predict": 64}
    )
    return resp["message"]["content"]


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_query(query: str, system_prompt: str) -> str:
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    try:
        t0 = time.perf_counter()
        query_embedding = embed_query(query)
        t1 = time.perf_counter()
 
        context = retrieve_context(driver, query_embedding)
        t2 = time.perf_counter()
        #context.print()  # optional: print the retrieved context for debugging
 
        answer = generate_answer(query, context, CHAT_MODEL, SYSTEM_QUERY_QA)
        t3 = time.perf_counter()
 
        embed_time = t1 - t0
        retrieve_time = t2 - t1
        generate_time = t3 - t2
 
        with open(LOG_PATH, "a") as f:
            f.write(f"{embed_time};{retrieve_time};{generate_time}\n")
 
        return answer
    finally:
        driver.close()

def load_txt(filename: str) -> list[dict]:
    with open(filename, "r", encoding="utf-8") as f:
        return [{"content": line.rstrip("\n"), "source": filename} for line in f]

def contradiction_check(filename):
    reqs_input = load_txt(filename)
    for req in reqs_input:
        print(f"\nInput requirement: {req['content']}")
        output = run_query(req['content'], SYSTEM_QUERY_CONTRADICTION)
        print(f"\nOutput: {output}\n")


def query_qa(filename: str):
    reqs_input = load_txt(filename)
    output_path = OUTPUT_PATH
    with open(output_path, "a", encoding="utf-8") as f:
        for req in reqs_input[3:]:
            print(f"\nInput requirement: {req['content']}")
            output = run_query(req['content'], system_prompt=SYSTEM_QUERY_QA)
            print(f"\nOutput: {output}\n")
            f.write(f"{output} <END>\n")

def main():
    parser = argparse.ArgumentParser(description="Query a Neo4j GraphRAG index via Ollama")
    parser.add_argument("query", nargs="*", help="Query text (omit to be prompted)")
    args = parser.parse_args()
    
    query = "Can any user of this system be NOT authorized?"
    if not query:
        print("No query provided.", file=sys.stderr)
        sys.exit(1)

    #answer = run_query(query)
    #print("\n=== Answer ===\n")
    #print(answer)
    #contradiction_check(filename)
    query_qa(INPUT_PATH)

if __name__ == "__main__":
    main()