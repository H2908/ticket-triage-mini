import chromadb
from chromadb.utils import embedding_functions

KB = [
    {"id":"kb-1","cat":"billing","title":"Billing & Payments",
     "content":"Invoices generated on 1st of each month. Download from Account > Billing. Dispute within 14 days. Late payment incurs 1.5% monthly fee. Contact billing@company.com."},
    {"id":"kb-2","cat":"order_management","title":"Order Tracking & Changes",
     "content":"Track orders via My Orders. Standard delivery 3-5 days. Express 1-2 days. Cancel within 1 hour of placing. After dispatch contact support to intercept."},
    {"id":"kb-3","cat":"returns_refunds","title":"Returns & Refunds",
     "content":"Returns accepted within 30 days. Go to Order History > Request Return. Refunds in 5-7 business days. Damaged items qualify for immediate replacement."},
    {"id":"kb-4","cat":"account_access","title":"Account Access & Security",
     "content":"Reset password via Forgot Password. Link valid 30 minutes. Enable 2FA under Account > Security. If hacked go to Security > Sign Out All Devices immediately."},
    {"id":"kb-5","cat":"technical_support","title":"Technical Troubleshooting",
     "content":"Clear cache and cookies first. Ensure app is updated. Compatible with iOS 14+ and Android 10+. Run Help > Diagnostics and share the report ID with support."},
    {"id":"kb-6","cat":"general_inquiry","title":"General Information",
     "content":"Support Monday-Friday 9am-6pm GMT. Ship to 45 countries. Free trial 14 days. Annual plan saves 20%. API available on Business plan."},
    {"id":"kb-7","cat":"urgent_escalation","title":"Urgent Escalation",
     "content":"Critical issues call 0800-XXX-XXXX (24/7) or email urgent@company.com. P1 SLA: response 15 mins, resolution 2 hours. Check status.company.com for outages."},
    {"id":"kb-8","cat":"feedback","title":"Feedback & Suggestions",
     "content":"Submit via Help > Send Feedback. Feature requests reviewed monthly. Complaints escalated within 2 business days. All feedback acknowledged within 24 hours."},
]

_kb = None

def get_kb():
    global _kb
    if _kb is not None:
        return _kb
    ef  = embedding_functions.SentenceTransformerEmbeddingFunction("all-MiniLM-L6-v2")
    cli = chromadb.Client()
    col = cli.create_collection("support_kb", embedding_function=ef,
                                metadata={"hnsw:space":"cosine"})
    col.add(
        documents=[a["content"] for a in KB],
        metadatas=[{"id":a["id"],"title":a["title"],"cat":a["cat"]} for a in KB],
        ids=[a["id"] for a in KB],
    )
    _kb = col
    print("KB ready — 8 articles indexed.")
    return _kb

def retrieve(query: str, k: int = 2) -> list[dict]:
    col     = get_kb()
    results = col.query(query_texts=[query], n_results=k,
                        include=["documents","metadatas","distances"])
    out = []
    for doc, meta, dist in zip(results["documents"][0],
                                results["metadatas"][0],
                                results["distances"][0]):
        out.append({"title":meta["title"],"content":doc,
                    "relevance":round(1-float(dist),3)})
    return out

def format_context(articles: list[dict]) -> str:
    if not articles:
        return ""
    lines = ["### Relevant Knowledge Base:"]
    for a in articles:
        lines.append(f"\n{a['title']} (relevance: {a['relevance']})")
        lines.append(a["content"])
    return "\n".join(lines) + "\n\n"

