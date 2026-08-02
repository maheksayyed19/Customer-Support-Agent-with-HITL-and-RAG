"""
Customer
      │
      ▼
classify_intent
      │
      ▼
retrieve_context
      │
      ▼
draft_response
      │
      ▼
check_confidence
      │
      ├───────────────┐
      │               │
Auto Resolve     interrupt()
      │               │
      ▼               ▼
      END      Human Review
                      │
          Approve / Edit / Reject
                      │
            ┌─────────┴─────────┐
            │                   │
        Resolve          Draft Again
            │                   │
            └─────────► Resolve
"""
"""
Customer Support Agent with Human-in-the-Loop (HITL), built on LangGraph.

Flow:
    classify_intent -> retrieve_context -> draft_response -> check_confidence
        -> (high confidence)  -> auto_resolve -> END
        -> (low confidence / sensitive intent) -> INTERRUPT (human_review)
            -> approved  -> resolve -> END
            -> edited    -> resolve (with human's text) -> END
            -> rejected  -> draft_response (revise, loop back)

Sensitive intents (refund, complaint) ALWAYS require human approval,
regardless of confidence score -- this is a business safety rule, not a
model judgement call.
"""
import os
import re
import json
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command
from groq import Groq

from state import SupportState
from database import lookup_order, init_db
import rag

init_db()
load_dotenv()
rag.ingest_documents()  # embeds docs/ into ChromaDB once; skips if already done  # reads .env in the project root into os.environ

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
MODEL = "llama-3.3-70b-versatile"

SENSITIVE_INTENTS = {"refund", "complaint"}
CONFIDENCE_THRESHOLD = 0.75

# Safety net: these phrases MUST always trigger human review, even if the LLM
# classifier mislabels the intent as something else (e.g. "other"). Business
# safety rules should not depend entirely on a model call being correct.
SENSITIVE_KEYWORDS = [
    "refund", "money back", "reimburse", "compensation",
    "terrible", "unhappy", "disappointed", "complaint", "complain",
    "damaged", "broken", "defective", "cancel my order",
]


def _llm(prompt: str, system: str = "") -> str:
    """Thin wrapper so the rest of the graph doesn't care about the provider."""
    if client is None:
        return "[LLM not configured - set GROQ_API_KEY]"
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=400,
    )
    return resp.choices[0].message.content.strip()


# ---------------------------------------------------------------------
# NODES
# ---------------------------------------------------------------------

def classify_intent(state: SupportState) -> SupportState:
    query = state["customer_query"]
    prompt = f"""Classify this customer support message into exactly one category:
faq, order_status, refund, complaint, other.

Message: "{query}"

Respond with ONLY the category word, nothing else."""
    raw = _llm(prompt, system="You are an intent classifier. Be precise.").lower().strip()
    intent = next((i for i in ["faq", "order_status", "refund", "complaint"] if i in raw), "other")

    history = state.get("history", [])
    history.append({"node": "classify_intent", "result": intent})
    return {**state, "intent": intent, "history": history}


def retrieve_context(state: SupportState) -> SupportState:
    query = state["customer_query"]
    intent = state["intent"]

    if intent == "order_status":
        match = re.search(r"ORD\d+", query.upper())
        context = lookup_order(match.group(0)) if match else "No order ID found in message."
    else:
        # faq / refund / complaint / other -> real semantic search over the docs/
        # folder (return policy, shipping policy, payments FAQ), not keyword matching.
        # This is what lets a customer say "package never showed up" and still
        # correctly retrieve the shipping policy, even with zero shared words.
        context = rag.retrieve(query, k=2)

    history = state.get("history", [])
    history.append({"node": "retrieve_context", "result": context})
    return {**state, "retrieved_context": context, "history": history}


def draft_response(state: SupportState) -> SupportState:
    query = state["customer_query"]
    intent = state["intent"]
    context = state.get("retrieved_context", "")
    rejection_note = ""
    if state.get("human_decision") == "rejected" and state.get("human_feedback"):
        rejection_note = f"\n\nA human reviewer rejected your previous draft with this feedback: \"{state['human_feedback']}\". Revise accordingly."

    prompt = f"""You are a customer support agent. Write a short, warm, professional reply.

Customer message: "{query}"
Detected intent: {intent}
Relevant context: {context}
{rejection_note}

Write ONLY the reply text, no preamble."""
    draft = _llm(prompt, system="You are a helpful, concise customer support agent.")

    # crude confidence heuristic: lower if context lookup failed, or model was unsure
    confidence = 0.9
    if "No relevant information found" in context or "No order found" in context or "not configured" in draft:
        confidence = 0.4

    history = state.get("history", [])
    history.append({"node": "draft_response", "result": draft, "confidence": confidence})
    revision_count = state.get("revision_count", 0) + (1 if state.get("human_decision") == "rejected" else 0)

    return {**state, "draft_response": draft, "confidence_score": confidence,
            "history": history, "revision_count": revision_count, "human_decision": None}


def check_confidence(state: SupportState) -> SupportState:
    intent = state["intent"]
    confidence = state.get("confidence_score", 0.0)
    query_lower = state["customer_query"].lower()

    keyword_flagged = any(kw in query_lower for kw in SENSITIVE_KEYWORDS)
    requires_approval = (intent in SENSITIVE_INTENTS) or (confidence < CONFIDENCE_THRESHOLD) or keyword_flagged

    history = state.get("history", [])
    history.append({"node": "check_confidence",
                     "result": f"requires_approval={requires_approval} "
                               f"(intent={intent}, conf={confidence}, keyword_flagged={keyword_flagged})"})
    print(f"!!! DEBUG check_confidence ran: intent={intent}, requires_approval={requires_approval} !!!")
    return {**state, "requires_approval": requires_approval, "history": history}

def route_after_confidence(state: SupportState) -> str:
    if state.get("revision_count", 0) >= 3:
        return "auto_resolve"  # safety valve: stop looping, force resolution
    return "human_review" if state.get("requires_approval") else "auto_resolve"


def auto_resolve(state: SupportState) -> SupportState:
    return {**state, "final_response": state["draft_response"], "status": "auto_resolved"}


def human_review(state: SupportState) -> SupportState:
    """
    This node PAUSES the graph using LangGraph's interrupt().
    Execution stops here until the caller resumes with a Command(resume=...)
    containing the human's decision.
    """
    decision_payload = interrupt({
        "ticket_id": state["ticket_id"],
        "customer_query": state["customer_query"],
        "draft_response": state["draft_response"],
        "confidence_score": state["confidence_score"],
        "intent": state["intent"],
    })
    # decision_payload = {"decision": "approved"|"edited"|"rejected", "text": "...", "reason": "..."}
    decision = decision_payload.get("decision")
    history = state.get("history", [])
    history.append({"node": "human_review", "result": decision})

    if decision == "approved":
        return {**state, "human_decision": "approved", "status": "awaiting_human", "history": history}
    elif decision == "edited":
        return {**state, "human_decision": "edited",
                "draft_response": decision_payload.get("text", state["draft_response"]),
                "status": "awaiting_human", "history": history}
    else:  # rejected
        return {**state, "human_decision": "rejected",
                "human_feedback": decision_payload.get("reason", ""),
                "status": "awaiting_human", "history": history}


def route_after_human(state: SupportState) -> str:
    decision = state.get("human_decision")
    if decision == "rejected":
        return "draft_response"  # loop back for revision
    return "resolve"


def resolve(state: SupportState) -> SupportState:
    return {**state, "final_response": state["draft_response"], "status": "resolved"}


# ---------------------------------------------------------------------
# GRAPH ASSEMBLY
# ---------------------------------------------------------------------

def build_graph():
    graph = StateGraph(SupportState)

    graph.add_node("classify_intent", classify_intent)
    graph.add_node("retrieve_context", retrieve_context)
    graph.add_node("draft_response", draft_response)
    graph.add_node("check_confidence", check_confidence)
    graph.add_node("auto_resolve", auto_resolve)
    graph.add_node("human_review", human_review)
    graph.add_node("resolve", resolve)

    graph.set_entry_point("classify_intent")
    graph.add_edge("classify_intent", "retrieve_context")
    graph.add_edge("retrieve_context", "draft_response")
    graph.add_edge("draft_response", "check_confidence")

    graph.add_conditional_edges("check_confidence", route_after_confidence, {
        "auto_resolve": "auto_resolve",
        "human_review": "human_review",
    })

    graph.add_conditional_edges("human_review", route_after_human, {
        "draft_response": "draft_response",
        "resolve": "resolve",
    })

    graph.add_edge("auto_resolve", END)
    graph.add_edge("resolve", END)

    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)


if __name__ == "__main__":
    app = build_graph()
    config = {"configurable": {"thread_id": "test-1"}}
    result = app.invoke({
        "ticket_id": "T1",
        "customer_name": "Test User",
        "customer_query": "What is your return policy?",
        "history": [],
        "revision_count": 0,
    }, config=config)
    print(json.dumps(result, indent=2, default=str))