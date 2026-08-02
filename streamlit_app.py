"""
Streamlit dashboard for the Customer Support Agent (HITL).

Two views:
  1. "Submit a Ticket" - simulates a customer sending a message
  2. "Human Review Queue" - where you approve / edit / reject the agent's drafts
"""
import streamlit as st
import requests

API_BASE = "http://localhost:8000"

st.set_page_config(page_title="Support Agent - HITL Dashboard", layout="wide")
st.title("🎧 Customer Support Agent — Human-in-the-Loop")

tab1, tab2, tab3 = st.tabs(["📨 Submit a Ticket", "🕵️ Human Review Queue", "📋 All Tickets"])

# ---------------------------------------------------------------
# TAB 1: Submit a ticket (simulates the customer side)
# ---------------------------------------------------------------
with tab1:
    st.subheader("Simulate an incoming customer message")
    with st.form("new_ticket"):
        name = st.text_input("Customer name", "Aisha Khan")
        query = st.text_area("Customer message",
                              "I want a refund for order ORD1001, it arrived damaged.")
        submitted = st.form_submit_button("Submit ticket")

    if submitted:
        resp = requests.post(f"{API_BASE}/ticket", json={
            "customer_name": name, "customer_query": query
        })
        if resp.ok:
            data = resp.json()
            if data["status"] == "awaiting_human":
                st.warning(f"Ticket {data['ticket_id']} needs human review. "
                           f"Check the 'Human Review Queue' tab.")
                st.json(data["pending_review"])
            else:
                st.success(f"Auto-resolved! Response sent:\n\n{data['final_response']}")
        else:
            st.error(f"Error: {resp.text}")

# ---------------------------------------------------------------
# TAB 2: Human review queue - the core HITL interaction
# ---------------------------------------------------------------
with tab2:
    st.subheader("Tickets awaiting your approval")
    if st.button("🔄 Refresh queue"):
        st.rerun()

    resp = requests.get(f"{API_BASE}/tickets/pending")
    pending = resp.json() if resp.ok else []

    if not pending:
        st.info("No tickets waiting for review right now.")

    for t in pending:
        with st.container(border=True):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"**Ticket `{t['ticket_id']}`** — {t['customer_name']}")
                st.markdown(f"*Customer said:* {t['customer_query']}")
                st.markdown(f"**Intent:** `{t['intent']}` &nbsp;&nbsp; "
                            f"**Confidence:** `{t['confidence_score']}`")
            with col2:
                st.metric("Confidence", f"{(t['confidence_score'] or 0)*100:.0f}%")

            edited_text = st.text_area("Agent's draft response (edit if needed)",
                                        t["draft_response"], key=f"draft_{t['ticket_id']}")

            b1, b2, b3 = st.columns(3)
            if b1.button("✅ Approve", key=f"approve_{t['ticket_id']}"):
                requests.post(f"{API_BASE}/ticket/{t['ticket_id']}/decision",
                              json={"decision": "approved"})
                st.rerun()

            if b2.button("✏️ Send edited version", key=f"edit_{t['ticket_id']}"):
                requests.post(f"{API_BASE}/ticket/{t['ticket_id']}/decision",
                              json={"decision": "edited", "text": edited_text})
                st.rerun()

            reason = st.text_input("Rejection reason (required to reject)",
                                    key=f"reason_{t['ticket_id']}")
            if b3.button("❌ Reject & revise", key=f"reject_{t['ticket_id']}"):
                if not reason:
                    st.error("Please provide a reason so the agent can revise.")
                else:
                    requests.post(f"{API_BASE}/ticket/{t['ticket_id']}/decision",
                                  json={"decision": "rejected", "reason": reason})
                    st.rerun()

# ---------------------------------------------------------------
# TAB 3: All tickets (simple dashboard)
# ---------------------------------------------------------------
with tab3:
    st.subheader("All tickets")
    resp = requests.get(f"{API_BASE}/tickets")
    all_t = resp.json() if resp.ok else []
    if all_t:
        st.dataframe(all_t, use_container_width=True)
    else:
        st.info("No tickets yet.")