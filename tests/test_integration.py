import os
from pathlib import Path

import pytest

from agent import main
from agent.mcp_client import call_mcp_tool

CHROMA_DIR = Path(__file__).resolve().parent.parent / "mcp_server" / "chroma_db"

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.getenv("TRIAGE_REAL_API_KEY") != "1" or not CHROMA_DIR.exists(),
        reason="needs OPENAI_API_KEY and a built mcp_server/chroma_db (run mcp_server/ingest.py)",
    ),
]


async def test_real_server_finds_relevant_policy():
    result = await call_mcp_tool("search_policy", {"query": "how to report a conflict of interest"})

    assert result["found"] is True
    assert result["results"][0]["policy_id"] == "POL-001"
    assert {"policy_id", "title", "excerpt", "source_file", "score"} <= set(result["results"][0])


async def test_real_server_returns_controlled_empty_result():
    result = await call_mcp_tool("search_policy", {"query": "how many vacation days do I get"})

    assert result == {"results": [], "found": False}


async def test_end_to_end_follow_up(config):
    state = await main.graph.ainvoke({
        "request": "I reported a possible conflict of interest two weeks ago, but I have not "
                   "received an update. Should I submit a new request?"
    }, config)
    result = state["result"]

    assert result.category == "existing_case_follow_up"
    assert result.recommended_action == "follow_up_existing_case"
    assert result.policy_sufficient is True
    assert "POL-002" in {reference.policy_id for reference in result.policy_references}


async def test_end_to_end_unsupported_request(config):
    state = await main.graph.ainvoke({"request": "How many vacation days do I get?"}, config)
    result = state["result"]

    assert result.policy_sufficient is False
    assert result.policy_references == []
    assert result.retrieval_error is None
