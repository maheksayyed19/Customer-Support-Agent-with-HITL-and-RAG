"""
FastAPI backend for the Customer Support Agent with HITL.

Endpoints:
    POST /ticket          -> create a new support ticket, runs the graph until
                             it either auto-resolves or pauses for human review
    GET  /tickets/pending -> list tickets awaiting human review
    POST /ticket/{id}/decision -> resume a paused ticket with a human decision
    GET  /tickets         -> list all tickets (for a simple dashboard)
"""
import uuid
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langgraph.types import Command

from agent_graph import build_graph
from database import save_ticket, get_pending_tickets, get_all_tickets

app = FastAPI(title="Customer Support Agent (HITL)")
graph = build_graph()

# in-memory map of ticket_id -> thread config, so we know how to resume each one
THREADS: dict = {}


class TicketRequest(BaseModel):
    customer_name: str
    customer_query: str


class DecisionRequest(BaseModel):
    decision: str          # "approved" | "edited" | "rejected"
    text: str | None = None    # required if edited
    reason: str | None = None  # required if rejected


@app.post("/ticket")
def create_ticket(req: TicketRequest):
    ticket_id = str(uuid.uuid4())[:8]
    config = {"configurable": {"thread_id": ticket_id}}
    THREADS[ticket_id] = config

    state = {
        "ticket_id": ticket_id,
        "customer_name": req.customer_name,
        "customer_query": req.customer_query,
        "history": [],
        "revision_count": 0,
    }
    result = graph.invoke(state, config=config)

    if "__interrupt__" in result:
        result["status"] = "awaiting_human"
        save_ticket(result)
        return {"ticket_id": ticket_id, "status": "awaiting_human",
                "pending_review": result["__interrupt__"][0].value}

    save_ticket(result)
    return {"ticket_id": ticket_id, "status": result.get("status"),
            "final_response": result.get("final_response")}


@app.get("/tickets/pending")
def pending_tickets():
    return get_pending_tickets()


@app.get("/tickets")
def all_tickets():
    return get_all_tickets()


@app.post("/ticket/{ticket_id}/decision")
def submit_decision(ticket_id: str, req: DecisionRequest):
    config = THREADS.get(ticket_id) or {"configurable": {"thread_id": ticket_id}}

    if req.decision == "edited" and not req.text:
        raise HTTPException(400, "text is required for an edited decision")
    if req.decision == "rejected" and not req.reason:
        raise HTTPException(400, "reason is required for a rejected decision")

    payload = {"decision": req.decision, "text": req.text, "reason": req.reason}
    result = graph.invoke(Command(resume=payload), config=config)

    if "__interrupt__" in result:
        # rejected -> revised -> paused again for another look
        result["status"] = "awaiting_human"
        save_ticket(result)
        return {"ticket_id": ticket_id, "status": "awaiting_human",
                "pending_review": result["__interrupt__"][0].value}

    save_ticket(result)
    return {"ticket_id": ticket_id, "status": result.get("status"),
            "final_response": result.get("final_response")}


@app.get("/")
def root():
    return {"message": "Customer Support Agent (HITL) API is running"}