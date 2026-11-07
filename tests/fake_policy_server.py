from mcp.server.mcpserver import MCPServer

mcp = MCPServer("fake-policy-server")

FOLLOW_UP_HIT = {
    "policy_id": "POL-002",
    "title": "Case Follow-Up and Status Update Policy",
    "excerpt": (
        "6. Should I submit a new request?\n"
        "Do not submit a new request for a concern that has already been submitted. "
        "Follow up on the existing case instead, even if you have not yet received an update."
    ),
    "source_file": "POL-002_case_follow_up_and_status_updates.pdf",
    "score": 0.62,
}

RAISING_CONCERN_HIT = {
    "policy_id": "POL-003",
    "title": "Raising a Concern",
    "excerpt": (
        "4. Information to include\n"
        "Required information: (a) a description of the concern; and "
        "(b) the approximate date or period when the incident happened."
    ),
    "source_file": "POL-003_raising_a_concern.pdf",
    "score": 0.41,
}


@mcp.tool()
def search_policy(query: str, k: int = 8) -> dict:
    """Return fixed policy search results for tests."""
    text = query.lower()
    if "trigger failure" in text:
        raise RuntimeError("search index unavailable")
    if "vacation" in text:
        return {"results": [], "found": False}
    return {"results": [RAISING_CONCERN_HIT, FOLLOW_UP_HIT][:k], "found": True}


if __name__ == "__main__":
    mcp.run()
