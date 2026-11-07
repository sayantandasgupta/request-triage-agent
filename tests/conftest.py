import os
import uuid
from pathlib import Path

import pytest
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")
os.environ["TRIAGE_REAL_API_KEY"] = "1" if os.getenv("OPENAI_API_KEY") else "0"
os.environ.setdefault("OPENAI_API_KEY", "sk-test-not-a-real-key")

from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from agent import main, mcp_client

FAKE_SERVER_PATH = Path(__file__).resolve().parent / "fake_policy_server.py"


class ToolCallingFakeModel(GenericFakeChatModel):
    """Fake chat model that accepts tools and replays scripted messages."""

    def bind_tools(self, tools, **kwargs):
        return self


class ScriptedCall:
    """Structured-output call that returns the next scripted output for one schema."""

    def __init__(self, model: "FakeModel", schema: type):
        self.model = model
        self.schema = schema

    async def ainvoke(self, *args, **kwargs):
        outputs = self.model.scripts.get(self.schema, [])
        if not outputs:
            raise AssertionError(f"Unexpected LLM call for {self.schema.__name__}")
        self.model.calls[self.schema] = self.model.calls.get(self.schema, 0) + 1
        return outputs.pop(0)


class FakeModel:
    """Fake chat model that returns scripted structured outputs per schema."""

    def __init__(self):
        self.scripts: dict[type, list] = {}
        self.calls: dict[type, int] = {}

    def script(self, schema: type, *outputs) -> None:
        """Queue outputs to return for a schema, in order."""
        self.scripts.setdefault(schema, []).extend(outputs)

    def with_structured_output(self, schema: type, **kwargs) -> ScriptedCall:
        return ScriptedCall(self, schema)


@pytest.fixture
def fake_model(monkeypatch) -> FakeModel:
    """Replace the agent's LLM with a scripted fake."""
    model = FakeModel()
    monkeypatch.setattr(main, "model", model)
    return model


@pytest.fixture
def fake_server(monkeypatch) -> Path:
    """Point the MCP client at the stand-in policy server."""
    monkeypatch.setattr(mcp_client, "SERVER_PATH", FAKE_SERVER_PATH)
    return FAKE_SERVER_PATH


@pytest.fixture
def search_agent(monkeypatch):
    """Install a search agent whose LLM calls search_policy with the given queries."""

    def install(*queries: str) -> None:
        messages = [
            AIMessage(content="", tool_calls=[
                {"name": "search_policy", "args": {"query": query}, "id": f"call-{number}"}])
            for number, query in enumerate(queries)
        ]
        messages.append(AIMessage(content="Search complete."))
        agent = create_agent(
            ToolCallingFakeModel(messages=iter(messages)),
            tools=[main.search_policy],
            system_prompt=main.SYSTEM_PROMPT,
        )
        monkeypatch.setattr(main, "triage_agent", agent)

    return install


@pytest.fixture
def config() -> dict:
    """Graph config with a fresh thread ID."""
    return {"configurable": {"thread_id": str(uuid.uuid4())}}
