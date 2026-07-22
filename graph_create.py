import asyncio
import logging
import time
import os
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_core.documents import Document
from langchain_neo4j import Neo4jGraph  
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("graph_log.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


class LoggingChatOllama(ChatOllama):
    def _log_prompt(self, prompt):
        logger.info("Sending prompt to Ollama: %s", repr(prompt) if not isinstance(prompt, str) else prompt)

    async def ainvoke(self, input, config=None, **kwargs):
        self._log_prompt(input)
        start = time.monotonic()
        result = await super().ainvoke(input, config=config, **kwargs)
        logger.info("Ollama responded in %.1fs", time.monotonic() - start)
        logger.info("Raw response content: %s", result.content)
        logger.info("Tool calls (if any): %s", getattr(result, "tool_calls", None))
        return result
        
def load_requirements(folder_path: str) -> list[Document]:
    docs = []
    for file_path in Path(folder_path).glob("test.txt"):
        text = file_path.read_text(encoding="utf-8")
        for i, line in enumerate(text.splitlines()):
            line = line.strip()
            if not line:
                continue
            docs.append(Document(
                page_content=line,
                metadata={
                    "source_file": file_path.name,
                    "requirement_id": f"{file_path.stem}-{i}",
                }
            ))
    return docs

llm = LoggingChatOllama(model="llama3.2", temperature=0)

# Constrain to your ontology - this is the important bit for your use case
graph_transformer = LLMGraphTransformer(
    llm=llm,
    allowed_nodes=["Requirement", "Actor", "Action", "Object", "Condition"],
    allowed_relationships=["PERFORMS", "ACTS_ON", "CONDITIONED_BY", "DEPENDS_ON"],
    strict_mode=False, 
)

docs = load_requirements("software_requirements_data")

load_dotenv()  # Load environment variables from .env file

graph = Neo4jGraph(
    url=os.getenv("NEO4J_URL"),
    username=os.getenv("NEO4J_USERNAME"),
    password=os.getenv("NEO4J_PASSWORD"),
)

async def get_graph_documents():
    logger.info("Starting graph document conversion for %d requirement document(s)", len(docs))
    if not docs:
        logger.warning("No requirement documents were loaded")
        return []

    try:
        graph_documents = await graph_transformer.aconvert_to_graph_documents(
            docs, config={"max_concurrency": 2}
        )
        total_nodes = sum(len(gd.nodes) for gd in graph_documents)
        total_rels = sum(len(gd.relationships) for gd in graph_documents)
        logger.info(
            "Graph conversion completed: %d document(s) -> %d node(s), %d relationship(s)",
            len(graph_documents), total_nodes, total_rels,
        )
        return graph_documents
    except Exception:
        logger.exception("Graph document conversion failed")
        raise

graph_documents = asyncio.run(get_graph_documents())

if graph_documents:
    graph.add_graph_documents(graph_documents, baseEntityLabel=True, include_source=True)

print(graph_documents[0].nodes)
print(graph_documents[0].relationships)