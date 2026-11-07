import textwrap

from agent.models.result import TriageResult

CATEGORY_LABELS = {
    "new_request": "New concern",
    "existing_case_follow_up": "Follow-up on an existing case",
    "policy_guidance": "Policy question",
    "other_or_unclear": "Unclear request",
}

ACTION_LABELS = {
    "submit_new_request": "Submit a new request",
    "follow_up_existing_case": "Follow up on your existing case",
    "provide_policy_guidance": "Policy guidance",
    "request_missing_information": "More information needed",
}


def shorten(text: str, width: int = 220) -> str:
    """Return the text on one line, cut to the given width."""
    return textwrap.shorten(" ".join(text.split()), width=width, placeholder=" ...")


def format_result(result: TriageResult) -> str:
    """Render a triage result as readable text for the user."""
    lines = [
        f"Request type: {CATEGORY_LABELS[result.category]}",
        f"Recommended next step: {ACTION_LABELS[result.recommended_action]}",
        "",
        result.guidance,
    ]

    if result.missing_information:
        lines += ["", "Information still needed:"]
        lines += [f"  - {item}" for item in result.missing_information]

    if result.policy_references:
        lines += ["", "Sources:"]
        for reference in result.policy_references:
            lines.append(
                f"  [{reference.policy_id}] {reference.title} ({reference.source_file})")
            lines.append(f"      \"{shorten(reference.excerpt)}\"")

    if result.retrieval_error:
        lines += ["", f"Error: {result.retrieval_error}"]

    lines += ["", result.advisory_notice]
    return "\n".join(lines)
