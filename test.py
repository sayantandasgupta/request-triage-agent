import asyncio

from agent.mcp_client import call_mcp_tool

QUERIES = [
    "how to report a conflict of interest",
    "follow up on a concern I submitted, no update",
    "accepting a dinner invitation from a vendor",
    "protection from retaliation after reporting",
    "how many vacation days do I get",
    "my laptop is broken",
    "where can I park my car",
]


async def main():
    for query in QUERIES:
        result = await call_mcp_tool("search_policy", {"query": query, "k": 3})
        top = [(hit["policy_id"], round(hit["score"], 3)) for hit in result["results"]]
        print(f"{query!r}: {top}")


asyncio.run(main())