import pytest

from agent import main
from agent.models.classification import Classification, ExtractedInfo
from agent.models.recommendation import Recommendation


@pytest.mark.parametrize("category, info, expected", [
    ("new_request", ExtractedInfo(concern_description="a conflict", incident_date_or_period="last week"), []),
    ("new_request", ExtractedInfo(concern_description="a conflict"),
     ["The approximate date or period of the incident"]),
    ("new_request", ExtractedInfo(),
     ["A description of the concern", "The approximate date or period of the incident"]),
    ("existing_case_follow_up", ExtractedInfo(case_reference="CR-2026-00417"), []),
    ("existing_case_follow_up", ExtractedInfo(original_submission_date="two weeks ago"), []),
    ("existing_case_follow_up", ExtractedInfo(),
     ["Your case or reference number, or the approximate date you submitted the original request"]),
    ("policy_guidance", ExtractedInfo(policy_topic="conflict of interest"), []),
    ("policy_guidance", ExtractedInfo(), ["The policy topic or question you are asking about"]),
])
def test_check_requirements(category, info, expected):
    state = {"category": category, "extracted_info": info}
    assert main.check_requirements(state)["missing_information"] == expected


def test_check_requirements_unclear_always_needs_clarification():
    state = {"category": "other_or_unclear", "extracted_info": ExtractedInfo()}
    assert main.check_requirements(state)["missing_information"]


def test_blank_strings_count_as_missing():
    info = ExtractedInfo(concern_description="a conflict", incident_date_or_period="  ")
    assert info.incident_date_or_period is None


@pytest.mark.parametrize("state, expected", [
    ({"category": "policy_guidance", "missing_information": []}, "search"),
    ({"category": "other_or_unclear", "missing_information": []}, "ask_user"),
    ({"category": "new_request", "missing_information": ["date"], "clarification_rounds": 2}, "ask_user"),
    ({"category": "new_request", "missing_information": ["date"], "clarification_rounds": 3}, "build_clarification"),
    ({"category": "other_or_unclear", "missing_information": ["x"], "clarification_rounds": 3}, "build_clarification"),
])
def test_route_after_check(state, expected):
    assert main.route_after_check(state) == expected


@pytest.mark.parametrize("state, expected", [
    ({"retrieval_error": "Policy search failed: Connection closed", "search_results": []}, "build_error_response"),
    ({"retrieval_error": None, "search_results": []}, "build_not_covered_response"),
    ({"retrieval_error": None, "search_results": [{"policy_id": "POL-001"}]}, "build_recommendation"),
])
def test_route_after_search(state, expected):
    assert main.route_after_search(state) == expected


def test_search_message_includes_category_and_follow_up_intent():
    state = {
        "request": "I reported a concern two weeks ago. Should I submit a new request?",
        "category": "existing_case_follow_up",
        "extracted_info": ExtractedInfo(
            original_submission_date="two weeks ago", follow_up_intent=["ask_whether_to_resubmit"]),
    }

    message = main.build_search_message(state)

    assert "Request category: existing_case_follow_up" in message
    assert "ask_whether_to_resubmit" in message
    assert state["request"] in message
    assert "case_reference" not in message


def test_describe_error_unwraps_exception_groups():
    error = ExceptionGroup("unhandled errors in a TaskGroup", [RuntimeError("Connection closed")])
    assert main.describe_error(error) == "Connection closed"


async def test_recommendation_drops_invented_citations(fake_model):
    fake_model.script(Recommendation, Recommendation(
        evidence_assessment="Cites an excerpt that does not exist.",
        policy_sufficient=True,
        recommended_action="provide_policy_guidance",
        guidance="Some guidance [POL-009].",
        cited_excerpt_ids=["E9"],
    ))
    state = {
        "request": "What does the policy say about gifts?",
        "category": "policy_guidance",
        "extracted_info": ExtractedInfo(policy_topic="gifts"),
        "search_results": [{
            "policy_id": "POL-004", "title": "Gifts and Hospitality Policy", "excerpt": "Gifts up to USD 50 ...",
            "source_file": "POL-004_gifts_and_hospitality.pdf", "score": 0.5,
        }],
    }

    output = await main.build_recommendation(state)

    assert output["policy_sufficient"] is False
    assert output["policy_references"] == []


async def test_recommendation_keeps_tool_excerpts(fake_model):
    fake_model.script(Recommendation, Recommendation(
        evidence_assessment="POL-004 answers the question.",
        policy_sufficient=True,
        recommended_action="provide_policy_guidance",
        guidance="Gifts up to USD 50 may be accepted [POL-004].",
        cited_excerpt_ids=["E1", "E1"],
    ))
    hit = {
        "policy_id": "POL-004", "title": "Gifts and Hospitality Policy", "excerpt": "Gifts up to USD 50 ...",
        "source_file": "POL-004_gifts_and_hospitality.pdf", "score": 0.5,
    }
    state = {
        "request": "What does the policy say about gifts?",
        "category": "policy_guidance",
        "extracted_info": ExtractedInfo(policy_topic="gifts"),
        "search_results": [hit],
    }

    output = await main.build_recommendation(state)

    assert output["policy_sufficient"] is True
    assert output["policy_references"] == [
        {key: hit[key] for key in ("policy_id", "title", "excerpt", "source_file")}]
