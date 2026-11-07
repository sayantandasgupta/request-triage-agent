import asyncio
import json
from typing import Literal
from dotenv import load_dotenv
from pathlib import Path
from pprint import pprint

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.messages import ToolMessage
from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command, interrupt

from agent.formatting import format_result
from agent.models.recommendation import Recommendation
from agent.models.requirements import RequirementsCheck
from agent.models.result import PolicyReference, TriageResult
from agent.prompts import CLASSIFY_PROMPT, RECOMMENDATION_PROMPT, SYSTEM_PROMPT

from agent.models.classification import Classification
from agent.models.state import TriageState
from agent.tools import search_policy

ROOT_DIR = Path(__file__).resolve().parent.parent
CATEGORY_ACTIONS = {
    "new_request": "submit_new_request",
    "existing_case_follow_up": "follow_up_existing_case",
    "policy_guidance": "provide_policy_guidance",
}

load_dotenv(ROOT_DIR / '.env')

model = init_chat_model(model="openai:gpt-4o-mini", temperature=0)

triage_agent = create_agent(
    model=model,
    tools=[search_policy],
    system_prompt=SYSTEM_PROMPT
)


async def classify(state: TriageState) -> TriageState:
    """Classify the user request into a set of selected categories"""
    result = await model.with_structured_output(Classification).ainvoke(
        CLASSIFY_PROMPT + f"```{state['request']}```"
    )

    return {"reasoning": result.reasoning, "category": result.category, "extracted_info": result.extracted_info}


def check_requirements(state: TriageState) -> TriageState:
    """List the required information missing for the request's category."""
    info = state["extracted_info"]
    category = state["category"]

    if category == "new_request":
        missing = []
        if not info.concern_description:
            missing.append("A description of the concern")
        if not info.incident_date_or_period:
            missing.append("The approximate date or period of the incident")
    elif category == "existing_case_follow_up":
        missing = [] if info.case_reference or info.original_submission_date else [
            "Your case or reference number, or the approximate date you submitted the original request"]
    elif category == "policy_guidance":
        missing = [] if info.policy_topic else [
            "The policy topic or question you are asking about"]
    else:
        missing = ["A clear description of what you need help with: a new concern, "
                   "a follow-up on an earlier concern, or a policy question"]

    return {"missing_information": missing}


def ask_user(state: TriageState) -> TriageState:
    """Ask the user for the missing information and add the answer to the request."""
    answer = interrupt({"missing_information": state["missing_information"]})

    return {
        "request": f"{state['request']}\nAdditional information from the user: {answer}",
        "clarification_rounds": state.get("clarification_rounds", 0) + 1,
    }


def describe_error(error: BaseException) -> str:
    """Return the message of the innermost exception."""
    while isinstance(error, BaseExceptionGroup) and error.exceptions:
        error = error.exceptions[0]
    return str(error) or type(error).__name__


def build_search_message(state: TriageState) -> str:
    """Build the search agent's input from the category, extracted information and request."""
    return (
        f"Request category: {state['category']}\n\n"
        f"Extracted information:\n{state['extracted_info'].model_dump_json(indent=2, exclude_none=True)}\n\n"
        f"Request:\n{state['request']}"
    )


async def search(state: TriageState) -> TriageState:
    """Search the policy documents through the triage agent."""
    try:
        result = await triage_agent.ainvoke(
            {"messages": [{"role": "user", "content": build_search_message(state)}]})
        tool_messages = [
            message for message in result["messages"]
            if isinstance(message, ToolMessage) and message.name == "search_policy"
        ]
        if not tool_messages:
            output = await search_policy.ainvoke({"query": state["request"]})
            tool_messages = [ToolMessage(content=json.dumps(
                output), name="search_policy", tool_call_id="direct")]
    except Exception as error:
        return {"search_results": [], "retrieval_error": f"Policy search failed: {describe_error(error)}"}

    failed = [message for message in tool_messages if message.status == "error"]
    if failed:
        return {"search_results": [], "retrieval_error": f"Policy search failed: {failed[0].content}"}

    results = {}
    for message in tool_messages:
        for hit in json.loads(message.content).get("results", []):
            results[(hit["policy_id"], hit["excerpt"])] = hit

    return {
        "search_results": sorted(results.values(), key=lambda hit: hit["score"], reverse=True),
        "retrieval_error": None,
    }


def build_clarification(state: TriageState) -> TriageState:
    """Build the clarification response for a request that is still missing information."""
    missing = state.get("missing_information") or []
    needed = "; ".join(
        missing) if missing else "more detail about what you need help with"

    if state["category"] == "other_or_unclear":
        guidance = (
            "The request could not be classified as a new concern, a follow-up on an earlier concern, "
            f"or a policy question. Please clarify: {needed}. "
            "If the request remains unclear, human review is recommended."
        )
    else:
        guidance = f"More information is needed before this request can be triaged. Please provide: {needed}."

    return {
        "recommended_action": "request_missing_information",
        "guidance": guidance,
        "policy_references": [],
        "policy_sufficient": False,
    }


async def build_recommendation(state: TriageState) -> TriageState:
    """Generate a recommendation grounded in the retrieved policy excerpts."""
    excerpts = {f"E{number}": hit for number,
                hit in enumerate(state["search_results"], start=1)}
    evidence = "\n\n".join(
        f"[{excerpt_id}] {hit['policy_id']} - {hit['title']} ({hit['source_file']})\n{hit['excerpt']}"
        for excerpt_id, hit in excerpts.items()
    )
    result = await model.with_structured_output(Recommendation).ainvoke([
        ("system", RECOMMENDATION_PROMPT),
        ("user",
         f"Request category: {state['category']}\n\n"
         f"Extracted information:\n{state['extracted_info'].model_dump_json(indent=2)}\n\n"
         f"Request:\n{state['request']}\n\n"
         f"Policy excerpts:\n{evidence}"),
    ])
    cited = [excerpts[excerpt_id] for excerpt_id in dict.fromkeys(
        result.cited_excerpt_ids) if excerpt_id in excerpts]
    if not result.policy_sufficient or not cited:
        return build_not_covered_response(state)
    return {
        "recommended_action": result.recommended_action,
        "guidance": result.guidance,
        "policy_references": [
            {key: hit[key] for key in ("policy_id", "title", "excerpt", "source_file")} for hit in cited
        ],
        "policy_sufficient": True,
    }


def build_error_response(state: TriageState) -> TriageState:
    """Build the response for a failed policy search."""
    return {
        "recommended_action": CATEGORY_ACTIONS[state["category"]],
        "guidance": (
            "The policy search could not be completed, so this request could not be checked "
            "against the policy documents and no policy guidance is given. "
            "Please try again later or ask for human review."
        ),
        "policy_references": [],
        "policy_sufficient": False,
    }


def build_not_covered_response(state: TriageState) -> TriageState:
    """Build the response for a request the policy documents do not cover."""
    return {
        "recommended_action": CATEGORY_ACTIONS[state["category"]],
        "guidance": (
            "The available policy documents do not contain guidance relevant to this request, "
            "so no policy-based recommendation can be given. Human review is recommended."
        ),
        "policy_references": [],
        "policy_sufficient": False,
    }


def route_after_check(state: TriageState) -> Literal["search", "ask_user", "build_clarification"]:
    if state["category"] != "other_or_unclear" and not state["missing_information"]:
        return "search"
    elif state.get("clarification_rounds", 0) < 3:
        return "ask_user"
    else:
        return "build_clarification"


def route_after_search(state: TriageState) -> Literal["build_recommendation", "build_not_covered_response", "build_error_response"]:
    if state["retrieval_error"]:
        return "build_error_response"
    elif not state["search_results"] or len(state["search_results"]) == 0:
        return "build_not_covered_response"
    else:
        return "build_recommendation"


def assemble_result(state: TriageState) -> TriageState:
    """Assemble the final structured output from the state."""
    return {"result": TriageResult(
        category=state["category"],
        extracted_info=state["extracted_info"],
        missing_information=state.get("missing_information") or [],
        recommended_action=state["recommended_action"],
        guidance=state["guidance"],
        policy_references=[PolicyReference(
            **reference) for reference in state.get("policy_references", [])],
        policy_sufficient=state.get("policy_sufficient", False),
        retrieval_error=state.get("retrieval_error"),
    )}


builder = StateGraph(TriageState)
builder.add_node("classify", classify)
builder.add_node("check_requirements", check_requirements)
builder.add_node("ask_user", ask_user)
builder.add_node("search", search)
builder.add_node("build_clarification", build_clarification)
builder.add_node("build_recommendation", build_recommendation)
builder.add_node("build_not_covered_response", build_not_covered_response)
builder.add_node("build_error_response", build_error_response)
builder.add_node("assemble_result", assemble_result)

builder.add_edge(START, "classify")
builder.add_edge("classify", "check_requirements")
builder.add_conditional_edges("check_requirements", route_after_check)
builder.add_edge("ask_user", "classify")
builder.add_conditional_edges("search", route_after_search)
builder.add_edge("build_clarification", "assemble_result")
builder.add_edge("build_recommendation", "assemble_result")
builder.add_edge("build_not_covered_response", "assemble_result")
builder.add_edge("build_error_response", "assemble_result")
builder.add_edge("assemble_result", END)

graph = builder.compile(checkpointer=InMemorySaver())


async def run(query: str):
    config = {"configurable": {"thread_id": "1"}}
    state = await graph.ainvoke(
        {"request": query},
        config
    )

    while "__interrupt__" in state:
        print("category:", state.get("category"))
        print("extracted_info:", state.get("extracted_info"))
        print("missing:", state.get("missing_information"))
        missing = state["__interrupt__"][0].value["missing_information"]
        answer = input(f"Please provide: {', '.join(missing)}\n> ")
        state = await graph.ainvoke(Command(resume=answer), config)

    return state

if __name__ == "__main__":
    final_state = asyncio.run(run(
        "I reported a possible conflict of interest two weeks ago, but I have not received an update. Should I submit a new request?"))
    print(format_result(final_state["result"]))
