from langchain_core.tools import tool

from agent.mcp_client import call_mcp_tool


@tool
async def search_policy(query: str) -> dict:
    """Search the synthetic policy documents for passages relevant to a query.

    Args:
        query: The policy topic or question to search for, in plain language.

    Returns:
        dict: A search result with two keys:
            results (list[dict]): Matching passages, most relevant first. Each has
                policy_id (str), title (str), excerpt (str), source_file (str) and score (float).
            found (bool): True if at least one passage was returned, otherwise False.
    """
    return await call_mcp_tool("search_policy", {"query": query})
