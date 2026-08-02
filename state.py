"""
State schema for the Customer Support Agent with HITL.
This is the single source of truth that flows through every node in the graph.
"""
from typing import TypedDict, Optional, Literal, List, Dict


class SupportState(TypedDict, total=False):
    # --- Input ---
    ticket_id: str
    customer_name: str
    customer_query: str

    # --- Reasoning trace ---
    intent: Optional[Literal["faq", "order_status", "refund", "complaint", "other"]]
    retrieved_context: Optional[str]        # KB or order data pulled in
    draft_response: Optional[str]           # what the agent proposes to send
    confidence_score: Optional[float]       # 0.0 - 1.0

    # --- HITL control ---
    requires_approval: bool                 # decided by check_confidence node
    human_decision: Optional[Literal["approved", "edited", "rejected"]]
    human_feedback: Optional[str]           # edited text OR rejection reason
    revision_count: int                     # guards against infinite loops

    # --- Output ---
    final_response: Optional[str]
    status: Literal["pending", "auto_resolved", "awaiting_human", "resolved", "rejected"]

    # --- Bookkeeping ---
    history: List[Dict[str, str]]           # simple audit log of what happened