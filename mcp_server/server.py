from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from mcp.server.mcpserver import MCPServer

BASE_DIR = Path(__file__).resolve().parent
CHROMA_DIR = BASE_DIR / "chroma_db"
ENV_PATH = BASE_DIR.parent / ".env"

# The MCP SDK v2 no longer reads .env files, so we load it ourselves
# (needed for OPENAI_API_KEY, which OpenAIEmbeddings reads from the environment).
load_dotenv(ENV_PATH)

mcp = MCPServer("Policy Server")

embedding = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma(
    persist_directory=str(CHROMA_DIR),
    collection_name="policies",
    embedding_function=embedding,
)


@mcp.tool()
def search_policy(query: str, k: int = 8) -> dict:
    """Search the synthetic policy documents for passages relevant to a query.

    Args:
        query: The policy topic or question to search for, in plain language.
        k: Maximum number of passages to return. Defaults to 8.

    Returns:
        dict: A search result with two keys:
            results (list[dict]): Matching passages, most relevant first. Each has
                policy_id (str), title (str), excerpt (str, the passage text),
                source_file (str, the PDF filename) and score (float, higher is more relevant).
            found (bool): True if at least one passage was returned, otherwise False.
    """
    hits = vectorstore.similarity_search_with_relevance_scores(query, k=k)
    results = [
        {
            "policy_id": d.metadata["policy_id"],
            "title": d.metadata["title"],
            "excerpt": d.page_content,
            "source_file": d.metadata["filename"],   # key written by ingest.py
            "score": score,
        }
        for d, score in hits
    ]
    return {"results": results, "found": len(results) > 0}


if __name__ == "__main__":
    mcp.run()