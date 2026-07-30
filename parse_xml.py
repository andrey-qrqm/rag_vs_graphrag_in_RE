"""
Chunking algorithm for nested requirements-engineering XML.

Hierarchy (by id dot-depth):
  depth 1  ->  header     (id="1")
  depth 2  ->  branch     (id="1.1")
  depth 3  ->  module     (id="1.1.1")
  depth 4  ->  usecase    (id="1.1.1.1")
  depth 5  ->  action OR requirement leaf (id="1.1.1.1.1")
  depth 6  ->  requirement, always a leaf (id="1.1.1.1.1.1")

A depth-5 node is a leaf requirement if it has no nested <p> children;
if it DOES have children, it's an "action" and its children (depth 6)
are the requirement leaves.

Output: one dict per requirement leaf, keys:
  [id, branch, module, usecase, action, requirement]
"""

import os
import xml.etree.ElementTree as ET
from pymongo import MongoClient
from pymongo.server_api import ServerApi
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()
MONGODB_USER = os.getenv("MONGODB_USER")
MONGODB_PASSWORD = os.getenv("MONGODB_PASSWORD")

MONGO_URI = f"mongodb+srv://{MONGODB_USER}:{MONGODB_PASSWORD}@slmsbt.pto3n3r.mongodb.net/"

MODEL_NAME = "sentence-transformers/multi-qa-mpnet-base-cos-v1"
FILENAME = "pure_dataset/XMLZIPFILE/2010-blitdraft.xml"

client = MongoClient(MONGO_URI, server_api=ServerApi('1'))
collection = client["rag_db"]["pure_blitdraft"]
 
 
def _depth(node_id: str) -> int:
    return node_id.count(".") + 1
 
 
def _get_text(elem: ET.Element, tag: str) -> str:
    child = elem.find(tag)
    if child is None:
        return ""
    return "".join(child.itertext()).strip()
 
 
def _parse(elem: ET.Element, context: dict, results: list) -> None:
    node_id = elem.get("id")
    if node_id is None:
        return
 
    depth = _depth(node_id)
    title = _get_text(elem, "title")
    text_body = _get_text(elem, "text_body")
    children = elem.findall("p")
 
    if depth == 1:
        ctx = {**context, "header": title}
        for child in children:
            _parse(child, ctx, results)
 
    elif depth == 2:
        ctx = {**context, "branch": title}
        for child in children:
            _parse(child, ctx, results)
 
    elif depth == 3:
        ctx = {**context, "module": title}
        for child in children:
            _parse(child, ctx, results)
 
    elif depth == 4:
        ctx = {**context, "usecase": title}
        for child in children:
            _parse(child, ctx, results)
 
    elif depth == 5:
        if children:
            # this node is an action; its children are requirement leaves
            ctx = {**context, "action": title}
            for child in children:
                _parse(child, ctx, results)
        else:
            # leaf requirement directly under a usecase, no action
            results.append({
                "id": node_id,
                "branch": context.get("branch"),
                "module": context.get("module"),
                "usecase": context.get("usecase"),
                "action": None,
                "requirement": text_body or title,
            })
 
    elif depth == 6:
        # always a requirement leaf, child of an action
        results.append({
            "id": node_id,
            "branch": context.get("branch"),
            "module": context.get("module"),
            "usecase": context.get("usecase"),
            "action": context.get("action"),
            "requirement": text_body or title,
        })
 
    else:
        # deeper than expected; fall back to treating it as a leaf requirement
        results.append({
            "id": node_id,
            "branch": context.get("branch"),
            "module": context.get("module"),
            "usecase": context.get("usecase"),
            "action": context.get("action"),
            "requirement": text_body or title,
        })
 
 
def _strip_namespaces(elem: ET.Element) -> None:
    """Remove '{namespace}' prefixes from every tag in the tree, in place."""
    for e in elem.iter():
        if "}" in e.tag:
            e.tag = e.tag.split("}", 1)[1]
 
 
def chunk_xml(path: str) -> list:
    """Parse the XML file at `path` and return a list of requirement dicts."""
    tree = ET.parse(path)
    root = tree.getroot()
    _strip_namespaces(root)
 
    # top-level <p> elements are the depth-1 headers.
    # handle both <root><p id="1">...</p></root> and root itself being <p id="1">
    top_level = root.findall("p") if root.tag != "p" else [root]
 
    results = []
    for node in top_level:
        _parse(node, {}, results)
    return results


def ingest():
    model = SentenceTransformer(MODEL_NAME)
    chunks = chunk_xml(FILENAME)
    texts = [r["requirement"] for r in chunks]
    embeddings = model.encode(texts, show_progress_bar=True)

    docs = [
        {"id": c["id"], 
         "branch": c["branch"],
         "module": c["module"],
         "usecase": c["usecase"],
         "action": c["action"],
         "requirement": c["requirement"],
         "embedding": emb.tolist()}
        for c, emb in zip(chunks, embeddings)
    ]

    collection.delete_many({})  # optional: clear old data before re-ingesting
    collection.insert_many(docs)
    print(f"Inserted {len(docs)} chunks")

 

if __name__ == "__main__":
    import sys
    import json
    FILENAME = "pure_dataset/XMLZIPFile/2010-blitdraft.xml"


    chunks = chunk_xml(FILENAME)
    print(json.dumps(chunks, indent=2, ensure_ascii=False))
    print(f"\n{len(chunks)} requirement chunks extracted.")
    ingest()
    print("Ingestion complete.")
    