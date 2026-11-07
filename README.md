# Request Triage Agent

A small, locally runnable AI agent that triages employee requests. It reads a request, classifies it,
extracts the information it contains, checks whether anything required is missing, searches a set of
synthetic policy documents through an MCP tool, and returns a structured, advisory recommendation that
cites the policies it is based on.

Built with Python, LangGraph / LangChain, the Model Context Protocol (MCP), OpenAI `gpt-4o-mini`, and a
Chroma vector index over five synthetic policy PDFs.

> All policy documents and test data are synthetic. Every recommendation is advisory only: it does not
> represent a final decision and takes no action on any real case.

---

## Contents

- [Quick start](#quick-start)
- [Project structure](#project-structure)
- [Workflow design](#workflow-design)
- [How requests are classified](#how-requests-are-classified)
- [How information is extracted and validated](#how-information-is-extracted-and-validated)
- [How missing information is identified](#how-missing-information-is-identified)
- [MCP server and policy search](#mcp-server-and-policy-search)
- [Synthetic policy documents](#synthetic-policy-documents)
- [Structured output](#structured-output)
- [Automated tests](#automated-tests)
- [Technical decisions](#technical-decisions)
- [Assumptions](#assumptions)
- [Limitations](#limitations)
- [AI-assistance declaration](#ai-assistance-declaration)

---

## Quick start

Requires Python 3.13 and an OpenAI API key.

```bash
# 1. Create a virtual environment and install dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Add your OpenAI API key (never commit this file; it is in .gitignore)
echo "OPENAI_API_KEY=sk-..." > .env

# 3. Build the policy vector index (mcp_server/chroma_db is generated, not committed)
python mcp_server/ingest.py

# 4. Run the agent on the example request from the assignment
python -m agent.main

# 5. Run the tests
pytest -v
```

Step 4 runs the example request from the assignment:

> "I reported a possible conflict of interest two weeks ago, but I have not received an update. Should I
> submit a new request?"

and prints:

```text
Request type: Follow-up on an existing case
Recommended next step: Follow up on your existing case

To follow up on your reported possible conflict of interest, use the Check Case Status option on the
Integrity Portal, or reply to your acknowledgement message. If you do not have a case reference number,
provide the approximate date when you submitted the concern and the channel you used for submission, so
the Ethics Office can locate your case. Do not submit a new request for this concern, as duplicate
submissions can slow down the review process. Follow up on the existing case instead, even if you have
not yet received an update. [POL-002]

Sources:
  [POL-002] Case Follow-Up and Status Update Policy (POL-002_case_follow_up_and_status_updates.pdf)
      "reporter. 4. How to request an update Use the Check Case Status option on the Integrity Portal, ..."
  [POL-002] Case Follow-Up and Status Update Policy (POL-002_case_follow_up_and_status_updates.pdf)
      "Follow up on the existing case instead, even if you have not yet received an update. Submit a new ..."
  [POL-002] Case Follow-Up and Status Update Policy (POL-002_case_follow_up_and_status_updates.pdf)
      "1. Purpose This policy explains how to follow up on a concern you have already submitted, ..."

This recommendation is advisory only. It does not represent a final decision and no action has been
taken on any actual case.
```

The wording of the guidance can vary slightly between runs, because it is written by the LLM.

If a request is missing required information, `python -m agent.main` prompts for it in the terminal
(see [How missing information is identified](#how-missing-information-is-identified)).

---

## Project structure

```text
agent/
  main.py              LangGraph workflow: nodes, routing, graph, and a small CLI runner
  prompts.py           Prompts for classification, policy search and recommendation
  tools.py             LangChain tool `search_policy`, which forwards to the MCP server
  mcp_client.py        Starts the MCP server over stdio and calls a tool on it
  formatting.py        Renders a TriageResult as readable text
  models/
    classification.py  Category values, ExtractedInfo, Classification (LLM output schema)
    recommendation.py  Allowed actions, Recommendation (LLM output schema)
    result.py          TriageResult (final structured output), PolicyReference, advisory notice
    state.py           TriageState shared between graph nodes
mcp_server/
  server.py            MCP server exposing one tool: search_policy
  ingest.py            Splits the policy PDFs into chunks, embeds them and builds the Chroma index
  documents/           Five synthetic policy PDFs
tests/
  conftest.py          Fake LLM, fake search agent, and fixtures
  fake_policy_server.py  Stand-in MCP server with fixed results, used by offline tests
  test_rules.py        Unit tests: requirement rules, routing, citation filtering
  test_scenarios.py    Workflow tests for the four required scenarios (no network)
  test_integration.py  Tests against the real LLM and the real MCP server
  test_results.txt     Output of the latest test run
```

---

## Workflow design

The agent is one LangGraph `StateGraph` (`agent/main.py`). All four categories go through the same
graph; the category decides which rules are applied, not which workflow runs.

```text
                 START
                   │
                   ▼
              ┌─────────┐
     ┌──────▶ │classify │  LLM: category + extracted information
     │        └────┬────┘
     │             ▼
     │   ┌──────────────────┐
     │   │check_requirements│  Python rules: what is missing for this category?
     │   └────────┬─────────┘
     │            │
     │   missing info or unclear?
     │      ├── yes, fewer than 3 rounds ──▶ ask_user (pause for the user's answer) ──┐
     │      ├── yes, 3 rounds used ────────▶ build_clarification ──────────────┐       │
     │      └── no ─────▶ search                                            │       │
     │                      │  search agent calls search_policy via MCP     │       │
     │        ┌─────────────┼─────────────────────┐                         │       │
     │        ▼             ▼                     ▼                         │       │
     │  build_error_   build_not_covered_   build_recommendation            │       │
     │  response       response             (LLM, grounded in excerpts)     │       │
     │  (search failed) (no results)               │                         │       │
     │        └─────────────┴──────────┬───────────┘                         │       │
     │                                 ▼                                     │       │
     │                         assemble_result ◀─────────────────────────────┘       │
     │                                 │                                             │
     │                                END                                            │
     └───────────────────────────────────────────────────────────────────────────────┘
```

| Node                         | Kind                   | What it does                                                                                                                  |
| ---------------------------- | ---------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| `classify`                   | LLM, structured output | Returns the category, a one-sentence reason, and the extracted information.                                                   |
| `check_requirements`         | Plain Python           | Applies the business rules for the category and lists what is missing.                                                        |
| `ask_user`                   | LangGraph `interrupt`  | Pauses the graph and asks the user for the missing items. Their answer is appended to the request, which is classified again. |
| `search`                     | LangChain agent + MCP  | A small tool-calling agent writes focused search queries and calls `search_policy` through MCP.                               |
| `build_recommendation`       | LLM, structured output | Writes guidance using only the retrieved excerpts, decides whether they are sufficient, and cites excerpt IDs.                |
| `build_clarification`        | Plain Python           | Returns `request_missing_information` with what is still needed.                                                              |
| `build_not_covered_response` | Plain Python           | Returns a "not covered by the policies" response with no guidance and recommends human review.                                |
| `build_error_response`       | Plain Python           | Returns a clear "policy search could not be completed" response with the error.                                               |
| `assemble_result`            | Plain Python           | Builds the final `TriageResult`.                                                                                              |

The graph is compiled with an in-memory checkpointer, so a paused run can be resumed with
`Command(resume=answer)` on the same `thread_id`.

---

## How requests are classified

`classify` calls `gpt-4o-mini` (temperature 0) with `with_structured_output(Classification)`, using
`CLASSIFY_PROMPT` in `agent/prompts.py`. The prompt:

- defines the four categories using the wording of the assignment;
- gives one example request per category, taken from the assignment;
- forbids inventing new categories;
- tells the model to treat any `Additional information from the user:` lines (added by `ask_user`) as
part of the same request.

The output schema (`agent/models/classification.py`) restricts `category` to a `Literal` of the four
allowed values, so the model cannot return anything else. The model also returns a one-sentence
`reasoning`, which is kept in the graph state for debugging.

`other_or_unclear` is used when no category fits reliably, for example "I need some help with a
workplace matter."

---

## How information is extracted and validated

Extraction happens in the same LLM call as classification. The model fills an `ExtractedInfo` object:

| Field                      | Used by                                    | Meaning                                                                                                           |
| -------------------------- | ------------------------------------------ | ----------------------------------------------------------------------------------------------------------------- |
| `concern_description`      | `new_request` (also filled for follow-ups) | What the concern is about, in the user's words                                                                    |
| `incident_date_or_period`  | `new_request`                              | When the incident happened, as stated (e.g. "last week")                                                          |
| `case_reference`           | `existing_case_follow_up`                  | An existing case or reference number                                                                              |
| `original_submission_date` | `existing_case_follow_up`                  | When the original concern was submitted (e.g. "two weeks ago")                                                    |
| `follow_up_intent`         | `existing_case_follow_up`                  | One or more of `request_update`, `provide_additional_information`, `ask_whether_to_resubmit`, `general_follow_up` |
| `policy_topic`             | `policy_guidance`                          | The policy topic or question asked about                                                                          |

Validation:

- Every field defaults to `None` (or an empty list). Each field's description and the prompt tell the
model to fill in only what the user actually said and not to guess.
- A Pydantic validator turns blank or whitespace-only strings into `None`, so an empty value from the
model always counts as missing.
- Pydantic validates the types, and the allowed values of `follow_up_intent`, on every LLM response.
- Dates are kept as the user stated them ("two weeks ago"). They are not converted to calendar dates.

---

## How missing information is identified

`check_requirements` applies the business rules from the assignment in plain Python, not with the LLM,
so the rules are applied the same way every time and can be unit tested:

| Category                  | Rule                                                                     | Missing item reported to the user                                                                                        |
| ------------------------- | ------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------ |
| `new_request`             | Needs **both** `concern_description` and `incident_date_or_period`       | "A description of the concern" and/or "The approximate date or period of the incident"                                   |
| `existing_case_follow_up` | Needs **at least one** of `case_reference` or `original_submission_date` | "Your case or reference number, or the approximate date you submitted the original request"                              |
| `policy_guidance`         | Needs `policy_topic`                                                     | "The policy topic or question you are asking about"                                                                      |
| `other_or_unclear`        | Always needs clarification                                               | "A clear description of what you need help with: a new concern, a follow-up on an earlier concern, or a policy question" |

What happens next:

1. If nothing is missing and the category is not `other_or_unclear`, the workflow goes on to the policy

   search.
2. Otherwise, the graph pauses with a LangGraph `interrupt` that lists the missing items. When the user

   answers, the answer is appended to the request and the request is classified and checked again. This
    demonstrates a few follow-up interactions, as the assignment allows.
3. After 3 rounds without the required information, the workflow stops asking and returns a structured

   result with `recommended_action = request_missing_information`, the missing items, and no policy
    references. For `other_or_unclear`, the guidance also recommends human review.

A request with missing information, or an unclear request, **never reaches the policy search**, so
unrelated policies are not searched.

---

## MCP server and policy search

### MCP server

`mcp_server/server.py` is an MCP server (official `mcp` Python SDK, stdio transport) that exposes one
tool:

```text
search_policy(query: str, k: int = 8) -> {"results": [...], "found": bool}
```

Each result contains `policy_id`, `title`, `excerpt`, `source_file` and `score` (higher is more
relevant). Results scoring below `0.3` are dropped. If nothing is left, the tool returns the controlled
empty result `{"results": [], "found": false}`.

### How the agent calls it

- `agent/mcp_client.py` starts the server as a subprocess over stdio, calls the tool and returns its
structured result. If the tool reports an error, it raises an exception that includes the error text.
- `agent/tools.py` wraps this as a LangChain tool, `search_policy`.
- The `search` node runs a small LangChain tool-calling agent (`create_agent`) with this tool. The agent
receives the request category, the extracted information and the request text. Its prompt tells it to
search for what the user needs to do next, not just the subject of the concern. For example, for
follow-ups it searches the follow-up process, with one search per follow-up intent. It may run up to 3
searches.
- Results from all searches are merged, duplicates are removed, and they are sorted by score.
- If the agent makes no tool call, `search` calls `search_policy` directly with the request text, so the
MCP tool is always invoked once the request passes the requirements check.

### Retrieval approach

Embeddings and vector search:

- `mcp_server/ingest.py` loads each PDF with `PyPDFLoader`.
- It strips the document header (title block and synthetic-document banner) from the first page.
- It splits the text into 1,000-character chunks with 200 characters of overlap.
- It embeds the chunks with OpenAI `text-embedding-3-small`.
- It stores them in a local Chroma collection that uses cosine similarity.
- The policy ID comes from the PDF's `Subject` metadata.

Chroma runs as a local folder (`mcp_server/chroma_db`), so there is no database server to run.

### Grounding the recommendation

`build_recommendation` gives the LLM the retrieved excerpts, each labelled `[E1]`, `[E2]`, … with its
policy ID, title and source file. Following `RECOMMENDATION_PROMPT`, the model:

- decides whether the excerpts answer the request **fully** (`policy_sufficient`), and gives no partial
guidance if they do not;
- uses only what the excerpts say, with no extra contacts, timelines, channels or steps;
- cites policy IDs such as `[POL-002]` after each point;
- does not promise or predict an outcome;
- lists the excerpt IDs its guidance is based on.

The code then checks the model's citations:

- Only excerpt IDs that were actually retrieved are kept. Invented IDs are dropped.
- If the model says the excerpts are insufficient, or no valid citation is left, the result becomes
"not covered" (no guidance, human review recommended) instead of uncited guidance.
- `policy_references` in the output are copied from the MCP tool's results, not from text written by the
LLM.

---

## Synthetic policy documents

Five fictional policies in `mcp_server/documents/`. Each is marked "SYNTHETIC DOCUMENT - fictional
guidance created for a technical prototype." They were written so that every test scenario can be shown.

| ID      | Title                                      | Main content                                                                                                              | Used for                         |
| ------- | ------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------- | -------------------------------- |
| POL-001 | Conflict of Interest Policy                | What a conflict of interest is, disclosing your own, reporting someone else's, what happens next                          | `new_request`, `policy_guidance` |
| POL-002 | Case Follow-Up and Status Update Policy    | Case reference format, timelines, requesting updates, adding information, **whether to submit a new request**, escalation | `existing_case_follow_up`        |
| POL-003 | Raising a Concern                          | Who can report, channels, **information to include** (description and date), anonymous concerns                           | `new_request`                    |
| POL-004 | Gifts and Hospitality Policy               | Gift value limits, prohibited gifts, hospitality                                                                          | `policy_guidance`                |
| POL-005 | Non-Retaliation and Confidentiality Policy | Protection from retaliation, confidentiality, anonymity                                                                   | All categories                   |

None of the policies cover topics such as leave or vacation, which is used to test the unsupported
request scenario.

---

## Structured output

The agent returns a `TriageResult` (Pydantic model, `agent/models/result.py`):

| Field                 | Type                                                                                                            | Meaning                                                                                               |
| --------------------- | --------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| `category`            | `new_request` \| `existing_case_follow_up` \| `policy_guidance` \| `other_or_unclear`                           | Classified request category                                                                           |
| `extracted_info`      | `ExtractedInfo`                                                                                                 | Information extracted from the request                                                                |
| `missing_information` | `list[str]`                                                                                                     | Required information that is still missing                                                            |
| `recommended_action`  | `provide_policy_guidance` \| `request_missing_information` \| `follow_up_existing_case` \| `submit_new_request` | Recommended next action                                                                               |
| `guidance`            | `str`                                                                                                           | Guidance or explanation supporting the recommendation, with `[POL-xxx]` citations                     |
| `policy_references`   | `list[PolicyReference]`                                                                                         | `policy_id`, `title`, `excerpt`, `source_file` for each cited excerpt, as returned by `search_policy` |
| `policy_sufficient`   | `bool`                                                                                                          | Whether the retrieved policy information was enough to answer the request                             |
| `retrieval_error`     | `str \| None`                                                                                                   | Error from the policy search, if it failed                                                            |
| `advisory_notice`     | `str`                                                                                                           | Fixed statement that the recommendation is advisory and not a final decision                          |

Example (abbreviated) for the request from the assignment:

```json
{
  "category": "existing_case_follow_up",
  "extracted_info": {
    "concern_description": "possible conflict of interest",
    "original_submission_date": "two weeks ago",
    "follow_up_intent": ["request_update", "ask_whether_to_resubmit"]
  },
  "missing_information": [],
  "recommended_action": "follow_up_existing_case",
  "guidance": "... Do not submit a new request for this concern ... Follow up on the existing case instead, even if you have not yet received an update. [POL-002]",
  "policy_references": [
    {
      "policy_id": "POL-002",
      "title": "Case Follow-Up and Status Update Policy",
      "excerpt": "Follow up on the existing case instead, even if you have not yet received an update. ...",
      "source_file": "POL-002_case_follow_up_and_status_updates.pdf"
    }
  ],
  "policy_sufficient": true,
  "retrieval_error": null,
  "advisory_notice": "This recommendation is advisory only. It does not represent a final decision and no action has been taken on any actual case."
}
```

How each ending fills the output:

| Situation                                            | `recommended_action`                                            | `policy_sufficient` | `policy_references` | `retrieval_error` |
| ---------------------------------------------------- | --------------------------------------------------------------- | ------------------- | ------------------- | ----------------- |
| Policies answer the request                          | Chosen by the LLM from the evidence, starting from the category | `true`              | Cited excerpts      | `null`            |
| Required information still missing / unclear request | `request_missing_information`                                   | `false`             | `[]`                | `null`            |
| No relevant policy, or excerpts not sufficient       | Default action for the category                                 | `false`             | `[]`                | `null`            |
| MCP server unavailable or tool error                 | Default action for the category                                 | `false`             | `[]`                | Error message     |

The default actions per category are `submit_new_request`, `follow_up_existing_case` and
`provide_policy_guidance`.

---

## Automated tests

Run with `pytest -v`. The latest results are in [`tests/test_results.txt`](tests/test_results.txt):
**34 passed** (macOS, Python 3.13.5, pytest 9.2.0).

There are three layers of tests:

| File                  | Tests | Network                    | What it covers                                                                                                                                                                                                                       |
| --------------------- | ----- | -------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `test_rules.py`       | 22    | None                       | Requirement rules for every category, blank-value handling, routing after the requirements check and after search, error-message unwrapping, search-agent input, citation filtering (invented citations dropped, duplicates removed) |
| `test_scenarios.py`   | 8     | None                       | The four required scenarios through the full graph. Uses a scripted fake LLM and a stand-in MCP server (`tests/fake_policy_server.py`) that runs as a real MCP subprocess, so the MCP client path is tested too                      |
| `test_integration.py` | 4     | OpenAI + real Chroma index | Real retrieval and the real end-to-end workflow. Skipped automatically if no `OPENAI_API_KEY` is set or the index has not been built                                                                                                 |

### Required scenarios

| Scenario                      | Offline tests                                                                                                                                                                                                                                                                                                                                                                                   | Integration tests                                                                                                                              |
| ----------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| **Relevant policy retrieval** | `test_relevant_policy_retrieval`: follow-up request returns `follow_up_existing_case`, cites POL-002 with title, excerpt and source file                                                                                                                                                                                                                                                        | `test_real_server_finds_relevant_policy` (top hit is POL-001 for a conflict-of-interest query), `test_end_to_end_follow_up` (POL-002 is cited) |
| **Missing information**       | `test_missing_information_asks_user_and_never_searches`: new request without a date asks the user, never searches, and returns `request_missing_information` after 3 rounds. `test_missing_information_continues_after_user_answers`: the workflow continues once the user gives the date. `test_unclear_request_is_never_searched`: unclear request recommends human review and never searches | none                                                                                                                                           |
| **Unsupported request**       | `test_unsupported_request_returns_not_covered`: no search results give "not covered" with no LLM recommendation call. `test_unsupported_request_when_excerpts_are_insufficient`: results exist but the LLM judges them insufficient                                                                                                                                                             | `test_real_server_returns_controlled_empty_result` (vacation query returns the controlled empty result), `test_end_to_end_unsupported_request` |
| **MCP failure**               | `test_mcp_server_unavailable`: server script missing. `test_mcp_tool_returns_error`: tool raises an error. Both return `retrieval_error`, a clear message, no references, and no LLM recommendation call                                                                                                                                                                                        | none                                                                                                                                           |

To run only the offline tests: `pytest -m "not integration"`.

---

## Technical decisions

- **LangGraph for the workflow, a LangChain agent only for search.** The overall flow is fixed and must
be predictable, so it is an explicit graph with conditional edges. Only the step that benefits from
the LLM choosing what to do, writing search queries, uses a tool-calling agent.
- **Business rules in code, not in the LLM.** The LLM extracts facts; Python decides what is missing.
This makes the required-information rules deterministic and unit-testable. An earlier LLM-based check
(`CHECK_REQUIREMENTS_PROMPT`, `RequirementsCheck`) is still in the code but not used.
- **Structured output everywhere.** Classification and recommendation use Pydantic schemas with
`Literal` types, so categories and actions are always valid values.
- **Grounding checked in code.** Citations are checked against the excerpts actually retrieved, and
policy references are copied from the tool output, so the LLM cannot invent a source.
- **Fail closed.** If evidence is missing, insufficient, or the search fails, the agent gives no policy
guidance and says so, instead of answering from general knowledge.
- **Search agent gets the category and extracted information.** Searching with the raw request tended
to retrieve policies about the _subject_ of the concern (e.g. conflict of interest) instead of the
_process_ the user asked about (e.g. whether to resubmit). Giving the search agent the category and
follow-up intent makes its queries target what the user needs to do next.
- **Embeddings with a local Chroma folder.** Semantic search handles paraphrased requests better than
keyword search, and Chroma needs no server. The relevance cutoff (0.3) gives the controlled empty
result for unrelated questions.
- `gpt-4o-mini` **at temperature 0.** Cheap and fast enough for a prototype, with reliable structured
output.
- **Offline tests with fakes.** A scripted fake LLM and a stand-in MCP server make the scenario tests
fast, free and deterministic, while still exercising the real graph and the real MCP client. A
separate integration suite checks the real system.

---

## Assumptions

- A request belongs to exactly one category.
- Dates and periods as stated by the user ("two weeks ago", "last week") are enough; they are not
checked against a calendar.
- A follow-up does not need a case reference if the approximate submission date is given, as in the
assignment's example.
- For `other_or_unclear`, asking the user a few times before returning the clarification result is
acceptable, since the assignment allows a few follow-up interactions.
- The policy documents are the only source of truth. Anything they do not cover is reported as not
covered, with human review recommended.

---

## Limitations

- **Missing information pauses the run instead of returning at once.** For a request with missing
information, the first call to the graph returns a LangGraph interrupt asking the user, not a
`TriageResult`. The structured clarification result is returned only after 3 rounds without the
required information, or the run continues once the information is given.
- **The CLI runs one fixed request.** `python -m agent.main` runs the example request from the
assignment. To try other requests, call `agent.main.run("...")` or `agent.main.graph` from Python.
- **LLM output varies.** Even at temperature 0, `gpt-4o-mini` does not always give the same answer, so
the guidance wording, and occasionally the sufficiency decision, can differ between runs. The offline
tests are deterministic; the integration tests can be affected.
- **One MCP server process per search.** Each `search_policy` call starts a new MCP server subprocess and
reopens the Chroma index, which takes about 2 seconds per search.
- **Fixed-size chunking.** Chunks are split by character count, not by policy section, so a chunk can
start mid-section or combine two sections.
- **Needs an OpenAI key and a built index.** The vector index is generated by `ingest.py` and is not
committed. The policy search and integration tests need network access to OpenAI.
- **In-memory state only.** Paused runs are kept in memory and are lost when the process exits.
- **No UI, API, authentication or persistence.** Out of scope for this prototype.

---

## AI-Assistance Declaration

I used **Claude** (Anthropic; Claude Sonnet 5.5 and Claude Opus 5.5, through the Claude app) as a learning aid and coding assistant throughout this project.

**What I used it for**

- **Learning:** explanations of MCP (servers, clients, tools), how LangChain's `create_agent` and LangGraph's `StateGraph` fit together, and checking current documentation and migration guides (MCP Python SDK v2, LangChain v1).
- **Synthetic data:** Claude generated the five fictional policy PDFs, designed so each test scenario can be demonstrated.
- **Code:** Claude wrote first drafts of several files and functions, including `ingest.py`, `server.py`, `mcp_client.py`, the Pydantic models, the search, recommendation and response-building nodes, the output formatter, and the automated test suite. I wrote the classification node, the graph wiring and the clarification loop, with Claude explaining the syntax. Claude also reviewed my code and helped debug errors.
- **Prompts:** I wrote the classification prompt. Claude drafted the search agent prompt and the recommendation prompt, and suggested edits to mine.

**Design decisions that were mine**

The workflow design and its trade-offs were decided by me, with Claude acting as a sounding board: using a LangGraph workflow rather than a single prompt-driven agent; asking the user for missing information and re-classifying; letting the LLM choose the recommended action; giving no partial guidance when evidence is insufficient; and citing specific excerpts.

**How I reviewed and validated the work**

- I ran every part locally and checked the outputs for one request of each category, plus unclear and unsupported requests.
- I tested the MCP server on its own, through Claude Desktop and a client script, before connecting it to the agent.
- I measured retrieval scores for covered and uncovered queries, and chose the score threshold from those results.
- When testing showed the LLM-based requirements check misapplying the business rules, I moved the rules into deterministic code.
- I read the generated tests, ran the full suite (results in `tests/test_results.txt`), and confirmed the tests fail when the routing logic is deliberately broken.
