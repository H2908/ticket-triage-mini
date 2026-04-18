import streamlit as st
import httpx, os

API = os.getenv("API_URL", "http://localhost:8000")
ICONS = {
    "billing":"💳","order_management":"📦","returns_refunds":"↩️",
    "account_access":"🔐","technical_support":"🔧",
    "general_inquiry":"💬","urgent_escalation":"🚨","feedback":"⭐",
}
PRIORITY = {
    "P1":("🔴","Critical"),"P2":("🟠","High"),
    "P3":("🟡","Medium"),"P4":("🟢","Low"),
}
SAMPLES = [
    "I was charged twice for my order. Please refund immediately.",
    "URGENT: Our system is down and we are losing money every minute.",
    "I cannot log into my account — password keeps being rejected.",
    "Your app crashes every time I open settings on iPhone 14.",
    "Your support team was absolutely brilliant today. Thank you.",
    "Where is my order? It was supposed to arrive yesterday.",
    "I want to return a jacket. It does not fit.",
    "Do you offer a student discount?",
]

st.set_page_config(page_title="AI Ticket Triage", page_icon="🎫", layout="wide")
st.title("🎫 AI Ticket Triage System")
st.caption("TinyLlama-1.1B · LoRA · ChromaDB RAG · FastAPI · Built by Harshit Taneja")

with st.sidebar:
    st.header("Sample Tickets")
    for s in SAMPLES:
        if st.button(s[:45]+"...", key=s, use_container_width=True):
            st.session_state["ticket"] = s
    st.divider()
    use_rag = st.toggle("Enable RAG", value=True)

ticket = st.text_area("Support Ticket",
                       value=st.session_state.get("ticket",""),
                       height=120,
                       placeholder="Type or paste a support ticket...",
                       key="ticket")
submit = st.button("🚀 Triage", type="primary")

if submit and ticket.strip():
    with st.spinner("Analysing..."):
        try:
            r = httpx.post(f"{API}/triage",
                           json={"ticket_text":ticket,"use_rag":use_rag},
                           timeout=60)
            r.raise_for_status()
            res = r.json()
            st.divider()
            c1,c2,c3,c4 = st.columns(4)
            icon         = ICONS.get(res["category"],"📝")
            p_icon,p_lbl = PRIORITY.get(res["priority"],("⚪","Unknown"))
            with c1: st.metric("Category", f"{icon} {res['category'].replace('_',' ').title()}")
            with c2: st.metric("Priority",  f"{p_icon} {p_lbl}")
            with c3: st.metric("Latency",   f"{res['latency_ms']:.0f}ms")
            with c4: st.metric("Status",    "Triaged ✓")
            st.divider()
            col1,col2 = st.columns(2)
            with col1:
                st.subheader("Summary")
                st.info(res["summary"])
            with col2:
                st.subheader("Suggested Action")
                st.success(res["suggested_action"])
            if use_rag and res.get("kb_articles"):
                st.divider()
                st.subheader("Knowledge Base Articles Retrieved")
                for a in res["kb_articles"]:
                    with st.expander(f"{a['title']} — relevance: {a['relevance']}"):
                        st.write(a["content"])
        except Exception as e:
            st.error(f"API error: {e}")
elif submit:
    st.warning("Please enter a ticket first.")

st.markdown("---")
st.caption("Harshit Taneja · MSc CS, University of Birmingham · github.com/H2908")
