# Customer Support Agent with Human-in-the-Loop (HITL) and RAG

An AI customer support agent built with **LangGraph**, **Groq (Llama 3.3)**, and **real semantic RAG** (ChromaDB + sentence-transformers). The agent handles routine queries autonomously, but pauses and hands control to a human reviewer for sensitive actions like refunds and complaints — before anything gets sent to a customer.

## 🔗 Live Demo

https://customer-support-agent-with-hitl-and-rag.streamlit.app/

> Note: the backend runs on a free tier and may take 30-60 seconds to wake up on the first request after inactivity.

## Why this project exists

Most "AI chatbot" projects auto-send whatever the model generates. This project demonstrates a more production-realistic pattern: **the agent proposes, a human disposes.** Sensitive requests are never auto-resolved, regardless of how confident the model is — because in a real business, a wrongly-issued refund or a mishandled complaint has real cost.

## Architecture

```
Customer message
      │
      ▼
classify_intent  ──►  detects: faq / order_status / refund / complaint / other
      │
      ▼
retrieve_context  ──►  semantic RAG search over company docs (ChromaDB)
      │                or order lookup (SQLite)
      ▼
draft_response   ──►  LLM drafts a reply grounded in retrieved context
      │
      ▼
check_confidence ──►  decides if a human must review, based on:
      │                 • sensitive intent (refund/complaint) — always
      │                 • low confidence (retrieval found nothing useful)
      │                 • keyword safety net (independent of LLM classification)
      │
   ┌──┴───────────────┐
   │                  │
auto_resolve      human_review  ──►  interrupt() pauses the graph here
   │                  │
   ▼                  ▼
  END          Approve / Edit / Reject
                      │
          ┌───────────┼────────────┐
          ▼           ▼            ▼
      resolve     resolve     draft_response
      (as-is)    (edited)     (revise & loop back)
          │           │
          └─────┬─────┘
                ▼
               END
```

## Key design decisions

**HITL is enforced two ways, not one.** Sensitive requests are gated both by LLM intent classification *and* an independent keyword safety net (`SENSITIVE_KEYWORDS` in `agent_graph.py`). This was a deliberate fix after testing revealed the LLM could occasionally misclassify a refund message as "other" — a business safety rule shouldn't depend entirely on a single model call being correct every time.

**RAG is real, not keyword matching.** Customer queries are embedded with a local, free sentence-transformer model (`all-MiniLM-L6-v2`) and matched against company policy docs in ChromaDB using semantic similarity — not string overlap. This means a query like *"my package never showed up"* correctly retrieves the shipping policy even though it shares almost no words with the source text.

**The pause/resume mechanism is LangGraph's `interrupt()` + checkpointer**, not a custom polling system. When `human_review` calls `interrupt()`, the entire graph execution freezes and its state is persisted by `MemorySaver`. Resuming later with `Command(resume=payload)` picks up exactly where it left off — including inside a loop, if a human rejects a draft multiple times.

## Tech stack

| Layer | Technology |
|---|---|
| Agent orchestration | LangGraph (`StateGraph`, `interrupt`, `Command`) |
| LLM | Groq (`llama-3.3-70b-versatile`) |
| RAG / embeddings | `sentence-transformers` (local, free) + ChromaDB |
| Backend API | FastAPI |
| Human review UI | Streamlit |
| Data | SQLite (tickets, mock orders) |

## Project structure

```
├── state.py           # LangGraph state schema (SupportState TypedDict)
├── agent_graph.py      # The agent: nodes, routing, HITL interrupt logic
├── rag.py               # Document chunking, embedding, ChromaDB retrieval
├── database.py          # Mock order DB + ticket persistence (SQLite)
├── main.py               # FastAPI endpoints wrapping the LangGraph app
├── streamlit_app.py       # Human review dashboard (submit / approve / edit / reject)
├── docs/                   # Company policy docs (return, shipping, payments) — the RAG knowledge base
├── requirements.txt
└── .env.example
```

## Running locally

**1. Clone and install**
```bash
git clone <this-repo-url>
cd customer-support-agent-with-hitl
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

**2. Set up your API key**
```bash
cp .env.example .env
# then edit .env and add your real GROQ_API_KEY (free at console.groq.com)
```

**3. (First run only) Verify RAG retrieval**
```bash
python rag.py
```
This embeds the docs in `docs/` and runs a few test queries — confirms semantic search is working before running the full app.

**4. Start the backend**
```bash
python -m uvicorn main:app
```

**5. Start the frontend** (in a second terminal, same folder)
```bash
streamlit run streamlit_app.py
```

**6. Try it**
- Submit a routine query like *"What is your return policy?"* → should auto-resolve
- Submit *"I want a refund, my order arrived damaged"* → should pause in the Human Review Queue tab
- Approve, edit, or reject the draft and watch it resolve or loop back for revision

## Known limitations (and what I'd change for production)

- **Confidence scoring is a simple heuristic**, not a calibrated model score — it only checks whether retrieval found something relevant. A production version would use the LLM's own log-probabilities or a dedicated scoring pass.
- **SQLite and ChromaDB are local files**, which don't persist on ephemeral hosting (e.g., Render's free tier). A production deployment would use a managed Postgres database and a hosted vector store.
- **Intent classification is a single LLM call** with no structured output guarantees — a production version might use function-calling / structured output modes to make parsing more robust.

## What I'd build next

- Structured output for intent classification instead of parsing free text
- A real calibrated confidence score
- Persistent storage for tickets and embeddings across restarts
- Multi-turn conversation support (currently each ticket is a single message)