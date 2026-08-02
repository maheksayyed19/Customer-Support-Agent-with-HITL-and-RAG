"""
Mock business data layer: FAQ knowledge base + order records + ticket persistence.
Swap this module out for a real DB/CRM/API when deploying for an actual client.
"""

import sqlite3
import json 
import os 
from datetime import datetime, timedelta

DB_PATH = os.path.join(os.path.dirname(__file__), "data", "support.db")

FAQS = [
    {"q": "What is your return policy?",
     "a": "Items can be returned within 30 days of delivery for a full refund, provided they are unused and in original packaging."},
    {"q": "How long does shipping take?",
     "a": "Standard shipping takes 3-5 business days. Express shipping takes 1-2 business days."},
    {"q": "Do you ship internationally?",
     "a": "Yes, we ship to over 40 countries. International orders take 7-14 business days."},
    {"q": "How do I track my order?",
     "a": "You can track your order using the tracking link sent to your email, or by logging into your account."},
    {"q": "What payment methods do you accept?",
     "a": "We accept credit/debit cards, UPI, net banking, and PayPal for international orders."},
]

MOCK_ORDERS = [
    {"order_id": "ORD1001", "customer": "Aisha Khan", "item": "Wireless Earbuds", "status": "Shipped",
     "amount": 1499, "days_ago": 2},
    {"order_id": "ORD1002", "customer": "Rahul Verma", "item": "Smart Watch", "status": "Delivered",
     "amount": 3499, "days_ago": 6},
    {"order_id": "ORD1003", "customer": "Priya Nair", "item": "Bluetooth Speaker", "status": "Processing",
     "amount": 1999, "days_ago": 1},
]
 
def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
 
    c.execute("""CREATE TABLE IF NOT EXISTS orders (
        order_id TEXT PRIMARY KEY,
        customer TEXT,
        item TEXT,
        status TEXT,
        amount REAL,
        order_date TEXT
    )""")
 
    c.execute("""CREATE TABLE IF NOT EXISTS tickets (
        ticket_id TEXT PRIMARY KEY,
        customer_name TEXT,
        customer_query TEXT,
        intent TEXT,
        draft_response TEXT,
        confidence_score REAL,
        requires_approval INTEGER,
        human_decision TEXT,
        human_feedback TEXT,
        final_response TEXT,
        status TEXT,
        created_at TEXT
    )""")
 
    c.execute("SELECT COUNT(*) FROM orders")
    if c.fetchone()[0] == 0:
        for o in MOCK_ORDERS:
            order_date = (datetime.now() - timedelta(days=o["days_ago"])).strftime("%Y-%m-%d")
            c.execute("INSERT INTO orders VALUES (?,?,?,?,?,?)",
                       (o["order_id"], o["customer"], o["item"], o["status"], o["amount"], order_date))
 
    conn.commit()
    conn.close()
 
 
def search_faq(query: str) -> str:
    """Very simple keyword-overlap search. Swap for embeddings/vector search at scale."""
    query_words = set(query.lower().split())
    best, best_score = None, 0
    for faq in FAQS:
        overlap = len(query_words & set(faq["q"].lower().split()))
        if overlap > best_score:
            best, best_score = faq, overlap
    if best and best_score > 0:
        return f"Q: {best['q']}\nA: {best['a']}"
    return "No matching FAQ found."
 
 
def lookup_order(order_id: str) -> str:
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT * FROM orders WHERE order_id = ?", (order_id,))
    row = c.fetchone()
    conn.close()
    if row:
        return (f"Order {row[0]}: {row[2]} for {row[1]}, status: {row[3]}, "
                f"amount: Rs.{row[4]}, ordered on {row[5]}")
    return f"No order found with ID {order_id}"
 
 
def save_ticket(state: dict):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""INSERT OR REPLACE INTO tickets VALUES
        (?,?,?,?,?,?,?,?,?,?,?,?)""",
        (state.get("ticket_id"), state.get("customer_name"), state.get("customer_query"),
         state.get("intent"), state.get("draft_response"), state.get("confidence_score"),
         int(state.get("requires_approval", False)), state.get("human_decision"),
         state.get("human_feedback"), state.get("final_response"), state.get("status"),
         datetime.now().isoformat()))
    conn.commit()
    conn.close()
 
 
def get_pending_tickets() -> list:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM tickets WHERE status = 'awaiting_human' ORDER BY created_at ASC")
    rows = [dict(r) for r in c.fetchall()]
    conn.close()
    return rows
 
 
def get_all_tickets() -> list:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM tickets ORDER BY created_at DESC")
    rows = [dict(r) for r in c.fetchall()]
    conn.close()
    return rows
 
 
if __name__ == "__main__":
    init_db()
    print("DB initialized at", DB_PATH)
    print(search_faq("what is your return policy"))
    print(lookup_order("ORD1001"))
 