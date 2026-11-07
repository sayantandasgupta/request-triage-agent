from langgraph.types import Command

from agent import main, mcp_client
from agent.models.classification import Classification, ExtractedInfo
from agent.models.recommendation import Recommendation
from agent.models.result import ADVISORY_NOTICE

FOLLOW_UP_REQUEST = (
    "I reported a possible conflict of interest two weeks ago, but I have not received an update. "
    "Should I submit a new request?"
)


def follow_up_classification() -> Classification:
    return Classification(
        reasoning="The user refers to a concern submitted two weeks ago.",
        category="existing_case_follow_up",
        extracted_info=ExtractedInfo(
            concern_description="possible conflict of interest",
            original_submission_date="two weeks ago",
            follow_up_intent=["request_update", "ask_whether_to_resubmit"],
        ),
    )


def new_request_without_date() -> Classification:
    return Classification(
        reasoning="The user wants to report a new concern.",
        category="new_request",
        extracted_info=ExtractedInfo(
            concern_description="possible conflict of interest in a procurement decision"),
    )


def vacation_question() -> Classification:
    return Classification(
        reasoning="The user asks a general policy question about vacation days.",
        category="policy_guidance",
        extracted_info=ExtractedInfo(policy_topic="vacation days"),
    )


async def test_relevant_policy_retrieval(fake_model, fake_server, search_agent, config):
    fake_model.script(Classification, follow_up_classification())
    fake_model.script(Recommendation, Recommendation(
        evidence_assessment="POL-002 says not to resubmit and to follow up on the existing case.",
        policy_sufficient=True,
        recommended_action="follow_up_existing_case",
        guidance="Do not submit a new request; follow up on your existing case instead [POL-002].",
        cited_excerpt_ids=["E1"],
    ))
    search_agent("following up on an existing concern")

    state = await main.graph.ainvoke({"request": FOLLOW_UP_REQUEST}, config)
    result = state["result"]

    assert result.category == "existing_case_follow_up"
    assert result.recommended_action == "follow_up_existing_case"
    assert result.policy_sufficient is True
    assert result.retrieval_error is None
    assert result.missing_information == []
    assert [reference.policy_id for reference in result.policy_references] == ["POL-002"]
    reference = result.policy_references[0]
    assert reference.title == "Case Follow-Up and Status Update Policy"
    assert reference.source_file == "POL-002_case_follow_up_and_status_updates.pdf"
    assert reference.excerpt.startswith("6. Should I submit a new request?")
    assert "[POL-002]" in result.guidance
    assert result.advisory_notice == ADVISORY_NOTICE


async def test_missing_information_asks_user_and_never_searches(fake_model, fake_server, search_agent, config):
    fake_model.script(Classification, *[new_request_without_date() for _ in range(4)])
    search_agent("conflict of interest")

    state = await main.graph.ainvoke(
        {"request": "I want to report a possible conflict of interest in a procurement decision."}, config)

    assert "__interrupt__" in state
    assert state["__interrupt__"][0].value["missing_information"] == [
        "The approximate date or period of the incident"]
    assert "search_results" not in state

    for answer in ["I am not sure", "no idea", "I can't remember"]:
        state = await main.graph.ainvoke(Command(resume=answer), config)

    result = state["result"]
    assert "__interrupt__" not in state
    assert result.recommended_action == "request_missing_information"
    assert result.missing_information == ["The approximate date or period of the incident"]
    assert result.policy_references == []
    assert result.policy_sufficient is False
    assert "search_results" not in state
    assert "Additional information from the user: I am not sure" in state["request"]
    assert fake_model.calls[Classification] == 4
    assert Recommendation not in fake_model.calls


async def test_missing_information_continues_after_user_answers(fake_model, fake_server, search_agent, config):
    fake_model.script(Classification, new_request_without_date(), Classification(
        reasoning="The user now gives the incident date.",
        category="new_request",
        extracted_info=ExtractedInfo(
            concern_description="possible conflict of interest in a procurement decision",
            incident_date_or_period="last week",
        ),
    ))
    fake_model.script(Recommendation, Recommendation(
        evidence_assessment="POL-003 explains what a new concern must include.",
        policy_sufficient=True,
        recommended_action="submit_new_request",
        guidance="Submit a new concern describing what happened and when [POL-003].",
        cited_excerpt_ids=["E2"],
    ))
    search_agent("how to report a new concern")

    state = await main.graph.ainvoke(
        {"request": "I want to report a possible conflict of interest in a procurement decision."}, config)
    assert "__interrupt__" in state

    state = await main.graph.ainvoke(Command(resume="It happened last week."), config)
    result = state["result"]

    assert result.recommended_action == "submit_new_request"
    assert result.policy_sufficient is True
    assert [reference.policy_id for reference in result.policy_references] == ["POL-003"]


async def test_unclear_request_is_never_searched(fake_model, fake_server, search_agent, config):
    unclear = Classification(
        reasoning="The request does not say what kind of help is needed.",
        category="other_or_unclear",
        extracted_info=ExtractedInfo(),
    )
    fake_model.script(Classification, *[unclear for _ in range(4)])
    search_agent("workplace matter")

    state = await main.graph.ainvoke({"request": "I need some help with a workplace matter."}, config)
    for answer in ["not sure", "it's complicated", "just help"]:
        state = await main.graph.ainvoke(Command(resume=answer), config)

    result = state["result"]
    assert result.category == "other_or_unclear"
    assert result.recommended_action == "request_missing_information"
    assert "human review" in result.guidance.lower()
    assert result.policy_references == []
    assert "search_results" not in state


async def test_unsupported_request_returns_not_covered(fake_model, fake_server, search_agent, config):
    fake_model.script(Classification, vacation_question())
    search_agent("vacation days")

    state = await main.graph.ainvoke({"request": "How many vacation days do I get?"}, config)
    result = state["result"]

    assert state["search_results"] == []
    assert result.category == "policy_guidance"
    assert result.policy_sufficient is False
    assert result.policy_references == []
    assert result.retrieval_error is None
    assert "do not contain guidance" in result.guidance
    assert Recommendation not in fake_model.calls
    assert result.advisory_notice == ADVISORY_NOTICE


async def test_unsupported_request_when_excerpts_are_insufficient(fake_model, fake_server, search_agent, config):
    fake_model.script(Classification, vacation_question())
    fake_model.script(Recommendation, Recommendation(
        evidence_assessment="The excerpts are about raising concerns, not leave.",
        policy_sufficient=False,
        recommended_action="provide_policy_guidance",
        guidance="",
        cited_excerpt_ids=[],
    ))
    search_agent("annual leave entitlement")

    state = await main.graph.ainvoke({"request": "How much annual leave do I get?"}, config)
    result = state["result"]

    assert state["search_results"]
    assert result.policy_sufficient is False
    assert result.policy_references == []
    assert "do not contain guidance" in result.guidance


async def test_mcp_server_unavailable(fake_model, monkeypatch, search_agent, config, tmp_path):
    monkeypatch.setattr(mcp_client, "SERVER_PATH", tmp_path / "missing_server.py")
    fake_model.script(Classification, follow_up_classification())
    search_agent("following up on an existing concern")

    state = await main.graph.ainvoke({"request": FOLLOW_UP_REQUEST}, config)
    result = state["result"]

    assert result.retrieval_error is not None
    assert result.retrieval_error.startswith("Policy search failed")
    assert result.policy_sufficient is False
    assert result.policy_references == []
    assert "could not be completed" in result.guidance
    assert Recommendation not in fake_model.calls
    assert result.advisory_notice == ADVISORY_NOTICE


async def test_mcp_tool_returns_error(fake_model, fake_server, search_agent, config):
    fake_model.script(Classification, follow_up_classification())
    search_agent("trigger failure")

    state = await main.graph.ainvoke({"request": FOLLOW_UP_REQUEST}, config)
    result = state["result"]

    assert result.retrieval_error is not None
    assert "search_policy" in result.retrieval_error
    assert "could not be completed" in result.guidance
    assert result.policy_sufficient is False
    assert result.policy_references == []
    assert Recommendation not in fake_model.calls
