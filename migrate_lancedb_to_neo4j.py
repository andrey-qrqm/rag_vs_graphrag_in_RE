"""
migrate_embeddings_lancedb_to_neo4j.py

One-time backfill: copy entity embeddings out of GraphRAG's LanceDB output
into the `embedding` property on the matching __Entity__ nodes in Neo4j.

GraphRAG writes embeddings to LanceDB (not to the Neo4j import) as part of
its output artifacts, typically under <graphrag_output_dir>/lancedb.
This script does NOT re-run embedding generation — it just moves vectors
that already exist from LanceDB into Neo4j.

Usage:
    python migrate_embeddings_lancedb_to_neo4j.py --list-tables
    python migrate_embeddings_lancedb_to_neo4j.py --table default-entity-description

Requires:
    pip install lancedb neo4j pandas
"""

import argparse

import lancedb
import pandas as pd
from neo4j import GraphDatabase
import os
from dotenv import load_dotenv

load_dotenv()

NEO4J_URI=os.getenv("NEO4J_URL")
NEO4J_USERNAME=os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD=os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE=os.getenv("NEO4J_DATABASE")

LANCEDB_PATH = "graphrag_pure/output/lancedb"   # adjust to your actual GraphRAG output path

BATCH_SIZE = 500
 
 
def _get_table_names() -> list[str]:
    """
    lancedb's list_tables()/table_names() API has proven unreliable across
    versions in this environment (e.g. routing through a namespace/catalog
    client that returns an empty ListTablesResponse even though local .lance
    folders exist on disk). Bypass the API and read table names directly
    from the filesystem instead — a lancedb local directory stores each
    table as a "<name>.lance" subfolder.
    """
    names = []
    for entry in os.listdir(LANCEDB_PATH):
        if entry.endswith(".lance"):
            names.append(entry[: -len(".lance")])
    if not names:
        raise FileNotFoundError(
            f"No .lance table folders found under {LANCEDB_PATH!r}. "
            "Check that LANCEDB_PATH points at the GraphRAG output/lancedb directory."
        )
    return sorted(names)
 
 
def list_tables():
    db = lancedb.connect(LANCEDB_PATH)
    for name in _get_table_names():
        tbl = db.open_table(name)
        print(f"{name}  (rows={tbl.count_rows()}, columns={tbl.schema.names})")
 
 
def load_embeddings(table_name: str) -> pd.DataFrame:
    db = lancedb.connect(LANCEDB_PATH)
    tbl = db.open_table(str(table_name))
    df = tbl.to_pandas()
 
    # LanceDB entity tables typically have an 'id' column matching the same
    # id used in entities.parquet / the Neo4j import, and a 'vector' column
    # holding the embedding. Print df.columns first if this doesn't match.
    if "id" not in df.columns or "vector" not in df.columns:
        raise ValueError(
            f"Expected 'id' and 'vector' columns, got: {list(df.columns)}. "
            "Inspect the table schema and adjust the column names below."
        )
    return df[["id", "vector"]]
 
 
def write_embeddings(df: pd.DataFrame):
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    rows = [
        {"id": row["id"], "embedding": list(row["vector"])}
        for _, row in df.iterrows()
    ]
 
    matched = 0
    with driver.session() as session:
        for i in range(0, len(rows), BATCH_SIZE):
            batch = rows[i : i + BATCH_SIZE]
            result = session.run(
                """
                UNWIND $rows AS row
                MATCH (e:__Entity__ {id: row.id})
                SET e.embedding = row.embedding
                RETURN count(e) AS updated
                """,
                rows=batch,
            ).single()
            matched += result["updated"]
            print(f"  batch {i // BATCH_SIZE + 1}: {result['updated']}/{len(batch)} matched")
 
    driver.close()
    print(f"\nTotal entities updated: {matched} / {len(rows)} rows in source table")
    if matched < len(rows):
        print(
            "WARNING: some ids did not match any __Entity__ node. "
            "Double check the id format matches what's in Neo4j "
            "(MATCH (e:__Entity__) RETURN e.id LIMIT 5 to compare)."
        )
 
 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--list-tables", action="store_true", help="List LanceDB tables and exit")
    parser.add_argument("--table", help="LanceDB table name to migrate (entity embeddings)")
    args = parser.parse_args()
 
    if args.list_tables:
        list_tables()
        return
 
    if not args.table:
        print("Pass --table <name>. Run --list-tables first to see available tables.")
        return
 
    df = load_embeddings(args.table)
    print(f"Loaded {len(df)} embedding rows from '{args.table}'")
    write_embeddings(df)
 
 
if __name__ == "__main__":
    main()
 