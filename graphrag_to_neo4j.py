import pandas as pd
from neo4j import GraphDatabase
from dotenv import load_dotenv
import time
import os

load_dotenv()
GRAPHRAG_FOLDER='./output'
NEO4J_URI=os.getenv("NEO4J_URL")
NEO4J_USERNAME=os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD=os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE=os.getenv("NEO4J_DATABASE")

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

def batched_import(statement, df, batch_size=1000):
    """
    Import a dataframe into Neo4j using a batched approach.
    Parameters: statement is the Cypher query to execute, df is the dataframe to import, and batch_size is the number of rows to import in each batch.
    """
    total = len(df)
    start_s = time.time()
    for start in range(0,total, batch_size):
        batch = df.iloc[start: min(start+batch_size,total)]
        result = driver.execute_query("UNWIND $rows AS value " + statement, 
                                      rows=batch.to_dict('records'),
                                      database_=NEO4J_DATABASE)
        print(result.summary.counters)
    print(f'{total} rows in { time.time() - start_s} s.')    
    return total


statements = """
create constraint chunk_id if not exists for (c:__Chunk__) require c.id is unique;
create constraint document_id if not exists for (d:__Document__) require d.id is unique;
create constraint entity_id if not exists for (c:__Community__) require c.community is unique;
create constraint entity_id if not exists for (e:__Entity__) require e.id is unique;
create constraint entity_title if not exists for (e:__Entity__) require e.name is unique;
create constraint entity_title if not exists for (e:__Covariate__) require e.title is unique;
create constraint related_id if not exists for ()-[rel:RELATED]->() require rel.id is unique;
""".split(";")

for statement in statements:
    if len((statement or "").strip()) > 0:
        print(statement)
        driver.execute_query(statement)

doc_df = pd.read_parquet(f'{GRAPHRAG_FOLDER}/documents.parquet', columns=["id", "title"])
print(doc_df.head(2))

statement = """
MERGE (d:__Document__ {id:value.id})
SET d += value {.title}
"""

batched_import(statement, doc_df)
text_df = pd.read_parquet(f'{GRAPHRAG_FOLDER}/text_units.parquet',
                          columns=["id","text","n_tokens","document_id"])
print(text_df.head(2))

statement = """
MERGE (c:__Chunk__ {id:value.id})
SET c += value {.text, .n_tokens}
WITH c, value
MATCH (d:__Document__ {id:value.document_id})
MERGE (c)-[:PART_OF]->(d)
"""

batched_import(statement, text_df)
entity_df = pd.read_parquet(f'{GRAPHRAG_FOLDER}/entities.parquet',
                            columns=["id","human_readable_id","title","type","description","text_unit_ids"])
print(entity_df.head(2))

entity_statement = """
MERGE (e:__Entity__ {id:value.id})
SET e += value {.human_readable_id, .description, name:replace(value.title,'"','')}
WITH e, value
CALL apoc.create.addLabels(e, case when coalesce(value.type,"") = "" then [] else [apoc.text.upperCamelCase(replace(value.type,'"',''))] end) yield node
UNWIND value.text_unit_ids AS text_unit
MATCH (c:__Chunk__ {id:text_unit})
MERGE (c)-[:HAS_ENTITY]->(e)
"""

batched_import(entity_statement, entity_df)

rel_df = pd.read_parquet(f'{GRAPHRAG_FOLDER}/relationships.parquet',
                         columns=['id', 'human_readable_id', 'source', 'target', 'description', 'weight', 'combined_degree', 'text_unit_ids'])
print(rel_df.head(2))

rel_statement = """
    MATCH (source:__Entity__ {name:replace(value.source,'"','')})
    MATCH (target:__Entity__ {name:replace(value.target,'"','')})
    MERGE (source)-[rel:RELATED {id: value.id}]->(target)
    SET rel += value {.combined_degree, .weight, .human_readable_id, .description, .text_unit_ids}
    RETURN count(*) as createdRels
"""


batched_import(rel_statement, rel_df)
community_df = pd.read_parquet(f'{GRAPHRAG_FOLDER}/communities.parquet', 
                     columns=['id', 'human_readable_id', 'community', 'level', 'parent', 'children', 'title', 'entity_ids', 'relationship_ids', 'text_unit_ids', 'period', 'size'])

print(community_df.head(2))

statement = """
MERGE (c:__Community__ {community:value.community})
SET c += value {.id, .human_readable_id, .level, .parent, .children, .title, .period, .size}
WITH *
UNWIND value.relationship_ids as rel_id
MATCH (start:__Entity__)-[:RELATED {id:rel_id}]->(end:__Entity__)
MERGE (start)-[:IN_COMMUNITY]->(c)
MERGE (end)-[:IN_COMMUNITY]->(c)
RETURN count(distinct c) as createdCommunities
"""

batched_import(statement, community_df)

community_report_df = pd.read_parquet(f'{GRAPHRAG_FOLDER}/community_reports.parquet',
                               columns=['id', 'human_readable_id', 'community', 'level', 'parent', 'children', 'title', 'summary', 'full_content', 'rank', 'rating_explanation', 'findings', 'full_content_json', 'period', 'size'])
community_report_df.head(2)


# import communities
community_statement = """
MERGE (c:__Community__ {community:value.community})
SET c += value {.level, .title, .rank, .rating_explanation, .full_content, .summary, .period, .size}
WITH c, value
UNWIND range(0, size(value.findings)-1) AS finding_idx
WITH c, value, finding_idx, value.findings[finding_idx] as finding
MERGE (c)-[:HAS_FINDING]->(f:Finding {id:finding_idx})
SET f += finding
"""
batched_import(community_statement, community_report_df)
