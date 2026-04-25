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
import logging
from typing import TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.graph import END, StateGraph

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LLM — initialised once at module level; reused across all nodes.
# Swap the model name here to change it everywhere simultaneously.
# ---------------------------------------------------------------------------

llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.85).bind(
    response_format={"type": "json_object"}
)

# ---------------------------------------------------------------------------
# Bot personas
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
    "crypto":  ["Bitcoin hits new all-time high amid regulatory ETF approvals",
                "Ethereum Layer-2 adoption surges 300% YoY"],
    "ai":      ["OpenAI releases GPT-5 with 10x reasoning improvements",
                "EU AI Act enforcement begins — fines up to €30M"],
    "market":  ["S&P 500 posts worst week since 2022 on recession fears",
                "Fed signals two more rate hikes in 2025"],
    "privacy": ["Meta fined $1.2B for illegal EU data transfers",
                "New browser fingerprinting technique bypasses all blockers"],
    "space":   ["SpaceX Starship completes first successful Mars trajectory test",
                "NASA Artemis III crew announced — Moon landing set for 2026"],
    "climate": ["Arctic sea ice hits record low for third consecutive year",
                "Renewable energy overtakes fossil fuels in EU grid mix"],
    "default": ["Tech stocks rally on better-than-expected earnings",
                "Global startup funding rebounds after two-year slump"],
}


@tool
def mock_searxng_search(query: str) -> str:
    """
    Simulates a SearXNG web search. Returns recent mock headlines
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
    final_output: dict  # the strict JSON deliverable


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
    system = (
        f"{state['persona']}\n\n"
        "You are deciding what to post on social media today. "
        "Pick ONE topic that strongly aligns with your worldview. "
        "Respond with ONLY a JSON object:\n"
        '{"topic": "<short topic label>", "search_query": "<4-8 word web search>"}'
    )

    response = llm.invoke([SystemMessage(content=system),
                           HumanMessage(content="What do you want to post about today?")])

    parsed = json.loads(response.content)
    logger.info("Node 1 — topic='%s' | query='%s'", parsed["topic"], parsed["search_query"])

    return {**state, "topic": parsed["topic"], "search_query": parsed["search_query"]}


def node_web_search(state: PostState) -> PostState:
    """
    Node 2 — Web Search
    Executes mock_searxng_search with the query produced by Node 1.
    """
    results = mock_searxng_search.invoke({"query": state["search_query"]})
    logger.info("Node 2 — search results:\n%s", results)
    return {**state, "search_results": results}


def node_draft_post(state: PostState) -> PostState:
    """
    Node 3 — Draft Post
    Generates a ≤280-char opinionated post using persona + search context.
    Enforces strict JSON output via Groq's json_object response format.
    """
    system = (
        f"{state['persona']}\n\n"
        "You write punchy, opinionated social-media posts (≤280 characters). "
        "You MUST respond with ONLY a JSON object:\n"
        '{"bot_id": "<bot_id>", "topic": "<topic>", "post_content": "<post ≤280 chars>"}\n\n'
        f"Your bot_id is: {state['bot_id']}\n"
        f"Today's topic: {state['topic']}\n"
        f"Real-world context (use this to make the post timely):\n{state['search_results']}"
    )

    response = llm.invoke([SystemMessage(content=system),
                           HumanMessage(content="Draft your post now.")])

    parsed = json.loads(response.content)
    parsed["post_content"] = parsed["post_content"][:280]  # hard cap at 280 chars

    logger.info("Node 3 — drafted post:\n%s", json.dumps(parsed, indent=2))
    return {**state, "post_content": parsed["post_content"], "final_output": parsed}


# ---------------------------------------------------------------------------
# 4. Graph factory
# ---------------------------------------------------------------------------

def build_content_graph() -> StateGraph:
    """
    Assembles and compiles the three-node LangGraph state machine.

    Flow: decide_search → web_search → draft_post → END
    """
    graph = StateGraph(PostState)

    graph.add_node("decide_search", node_decide_search)
    graph.add_node("web_search",    node_web_search)
    graph.add_node("draft_post",    node_draft_post)

    graph.set_entry_point("decide_search")
    graph.add_edge("decide_search", "web_search")
    graph.add_edge("web_search",    "draft_post")
    graph.add_edge("draft_post",    END)

    return graph.compile()