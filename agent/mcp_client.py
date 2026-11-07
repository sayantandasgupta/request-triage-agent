import json
import sys
from pathlib import Path

from mcp import Client, StdioServerParameters

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
SERVER_DIR = ROOT_DIR / "mcp_server"
SERVER_PATH = SERVER_DIR / "server.py"


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
