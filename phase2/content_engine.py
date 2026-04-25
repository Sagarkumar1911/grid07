"""
Phase 2: The Autonomous Content Engine (LangGraph)
===================================================
A three-node LangGraph state machine that:
  Node 1 — Decide Search  : LLM picks a topic given its persona
  Node 2 — Web Search     : calls mock_searxng_search tool
  Node 3 — Draft Post     : generates a 280-char opinionated post as strict JSON

Output schema: {"bot_id": "...", "topic": "...", "post_content": "..."}
"""

import json
import os
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.graph import END, StateGraph

load_dotenv()

# ---------------------------------------------------------------------------
# Bot personas (reused from Phase 1 for consistency)
# ---------------------------------------------------------------------------

BOT_PERSONAS = {
    "bot_a": (
        "You are Bot A — the Tech Maximalist. You believe AI and crypto will solve all human "
        "problems. You are highly optimistic about technology, Elon Musk, and space exploration. "
        "You dismiss regulatory concerns with data and enthusiasm."
    ),
    "bot_b": (
        "You are Bot B — the Doomer/Skeptic. You believe late-stage capitalism and tech "
        "monopolies are destroying society. You are highly critical of AI, social media, and "
        "billionaires. You value privacy and nature above profit."
    ),
    "bot_c": (
        "You are Bot C — the Finance Bro. You strictly care about markets, interest rates, "
        "trading algorithms, and making money. You speak in finance jargon and view everything "
        "through the lens of ROI and alpha generation."
    ),
}

# ---------------------------------------------------------------------------
# 1. Mock Tool — simulates SearXNG web search
# ---------------------------------------------------------------------------

MOCK_NEWS: dict[str, list[str]] = {
    "crypto":    ["Bitcoin hits new all-time high amid regulatory ETF approvals",
                  "Ethereum Layer-2 adoption surges 300% YoY"],
    "ai":        ["OpenAI releases GPT-5 with 10x reasoning improvements",
                  "EU AI Act enforcement begins — fines up to €30M"],
    "market":    ["S&P 500 posts worst week since 2022 on recession fears",
                  "Fed signals two more rate hikes in 2025"],
    "privacy":   ["Meta fined $1.2B for illegal EU data transfers",
                  "New browser fingerprinting technique bypasses all blockers"],
    "space":     ["SpaceX Starship completes first successful Mars trajectory test",
                  "NASA Artemis III crew announced — Moon landing set for 2026"],
    "climate":   ["Arctic sea ice hits record low for third consecutive year",
                  "Renewable energy overtakes fossil fuels in EU grid mix"],
    "default":   ["Tech stocks rally on better-than-expected earnings",
                  "Global startup funding rebounds after two-year slump"],
}


@tool
def mock_searxng_search(query: str) -> str:
    """
    Simulates a SearXNG web search.  Returns recent mock headlines
    based on keywords found in *query*.

    Parameters
    ----------
    query : search query string

    Returns
    -------
    Newline-separated headline strings
    """
    query_lower = query.lower()
    results: list[str] = []

    for keyword, headlines in MOCK_NEWS.items():
        if keyword in query_lower:
            results.extend(headlines)

    if not results:
        results = MOCK_NEWS["default"]

    return "\n".join(f"• {h}" for h in results[:3])


# ---------------------------------------------------------------------------
# 2. LangGraph State
# ---------------------------------------------------------------------------

class PostState(TypedDict):
    bot_id: str
    persona: str
    topic: str
    search_query: str
    search_results: str
    post_content: str
    final_output: dict   # the strict JSON deliverable


# ---------------------------------------------------------------------------
# 3. Node implementations
# ---------------------------------------------------------------------------

def node_decide_search(state: PostState) -> PostState:
    """
    Node 1 — Decide Search
    The LLM reads the bot's persona and decides:
      (a) what topic to post about today
      (b) what search query to fire
    """
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.9)

    system = (
        f"{state['persona']}\n\n"
        "You are deciding what to post on social media today. "
        "Pick ONE topic that strongly aligns with your worldview. "
        "Respond with ONLY a JSON object — no markdown fences:\n"
        '{"topic": "<short topic label>", "search_query": "<4-8 word web search>"}'
    )

    response = llm.invoke([SystemMessage(content=system),
                           HumanMessage(content="What do you want to post about today?")])

    raw = response.content.strip()
    # Strip accidental markdown fences
    raw = raw.replace("```json", "").replace("```", "").strip()
    parsed = json.loads(raw)

    print(f"[Node 1] Decided → topic='{parsed['topic']}' | query='{parsed['search_query']}'")

    return {**state, "topic": parsed["topic"], "search_query": parsed["search_query"]}


def node_web_search(state: PostState) -> PostState:
    """
    Node 2 — Web Search
    Executes mock_searxng_search with the query from Node 1.
    """
    results = mock_searxng_search.invoke({"query": state["search_query"]})
    print(f"[Node 2] Search results:\n{results}")
    return {**state, "search_results": results}


def node_draft_post(state: PostState) -> PostState:
    """
    Node 3 — Draft Post
    Generates a ≤280-char opinionated post; enforces strict JSON output.
    """
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.85)

    system = (
        f"{state['persona']}\n\n"
        "You write punchy, opinionated social-media posts (≤280 characters). "
        "You MUST respond with ONLY a JSON object — no markdown, no preamble:\n"
        '{"bot_id": "<bot_id>", "topic": "<topic>", "post_content": "<post ≤280 chars>"}\n\n'
        f"Your bot_id is: {state['bot_id']}\n"
        f"Today's topic: {state['topic']}\n"
        f"Real-world context (use this to make the post timely):\n{state['search_results']}"
    )

    response = llm.invoke([SystemMessage(content=system),
                           HumanMessage(content="Draft your post now.")])

    raw = response.content.strip().replace("```json", "").replace("```", "").strip()
    parsed = json.loads(raw)

    # Trim post_content to 280 chars just in case
    parsed["post_content"] = parsed["post_content"][:280]

    print(f"[Node 3] Drafted post:\n  {json.dumps(parsed, indent=2)}")
    return {**state, "post_content": parsed["post_content"], "final_output": parsed}


# ---------------------------------------------------------------------------
# 4. Assemble the LangGraph
# ---------------------------------------------------------------------------

def build_content_graph() -> StateGraph:
    graph = StateGraph(PostState)

    graph.add_node("decide_search", node_decide_search)
    graph.add_node("web_search",    node_web_search)
    graph.add_node("draft_post",    node_draft_post)

    graph.set_entry_point("decide_search")
    graph.add_edge("decide_search", "web_search")
    graph.add_edge("web_search",    "draft_post")
    graph.add_edge("draft_post",    END)

    return graph.compile()


# ---------------------------------------------------------------------------
# 5. Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("  Phase 2 — Autonomous Content Engine (LangGraph)")
    print("=" * 60)

    app = build_content_graph()

    for bot_id, persona in BOT_PERSONAS.items():
        print(f"\n{'─'*50}")
        print(f"Running graph for {bot_id} …")
        print(f"{'─'*50}")

        initial_state: PostState = {
            "bot_id":        bot_id,
            "persona":       persona,
            "topic":         "",
            "search_query":  "",
            "search_results": "",
            "post_content":  "",
            "final_output":  {},
        }

        result = app.invoke(initial_state)

        print(f"\n✅ Final JSON output for {bot_id}:")
        print(json.dumps(result["final_output"], indent=2))