import asyncio
import json
import sys
from pathlib import Path

from mcp import Client, StdioServerParameters

SERVER_PATH = Path(__file__).resolve().parent / "server.py"


async def call_mcp_tool(tool_name: str, arguments: dict) -> dict:
    params = StdioServerParameters(
        command=sys.executable, args=[str(SERVER_PATH)])

    async with Client(params) as client:
        result = await client.call_tool(tool_name, arguments)

        if result.is_error:
            raise RuntimeError(
                f"Tool '{tool_name}' failed: {result.content[0].text}")

        if result.structured_content is not None:
            return result.structured_content
        return json.loads(result.content[0].text)


if __name__ == "__main__":
    output = asyncio.run(call_mcp_tool(
        "search_policy",
        {"query": "I reported a possible conflict of interest two weeks ago, but I have "
                  "not received an update. Should I submit a new request?"},
    ))
    print(json.dumps(output, indent=2))
