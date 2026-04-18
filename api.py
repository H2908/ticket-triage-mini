import json, time, os
import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer
from rag import retrieve, format_context

MODEL_PATH = os.getenv("MODEL_PATH", "/content/model/final")
CATEGORIES = [
    "billing","order_management","returns_refunds","account_access",
    "technical_support","general_inquiry","urgent_escalation","feedback",
]
PROMPT = """### Support Ticket:
{ticket}

{context}### Instructions:
Classify and return ONLY valid JSON with keys: category, priority, summary, suggested_action.
Categories: {cats}

### Response:
"""

class TicketRequest(BaseModel):
    ticket_text: str = Field(..., min_length=5, max_length=500,
                             example="I cannot log into my account.")
    use_rag: bool = Field(True)

class TriageResult(BaseModel):
    category:         str
    priority:         str
    summary:          str
    suggested_action: str
    latency_ms:       float
    kb_articles:      list[dict]

print(f"Loading model from {MODEL_PATH}...")
_tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
_tokenizer.pad_token = _tokenizer.eos_token
_model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, torch_dtype=torch.float16, device_map="auto"
)
_model.eval()
print("Model loaded.")

def infer(ticket_text: str, use_rag: bool = True) -> dict:
    articles = retrieve(ticket_text, k=2) if use_rag else []
    context  = format_context(articles)
    prompt   = PROMPT.format(ticket=ticket_text, context=context,
                              cats=", ".join(CATEGORIES))
    inputs = _tokenizer(prompt, return_tensors="pt",
                        max_length=256, truncation=True).to(_model.device)
    t0 = time.perf_counter()
    with torch.no_grad():
        out = _model.generate(
            **inputs, max_new_tokens=120, temperature=0.1,
            do_sample=False, pad_token_id=_tokenizer.eos_token_id,
        )
    latency   = (time.perf_counter() - t0) * 1000
    generated = _tokenizer.decode(out[0], skip_special_tokens=True)
    response  = generated[len(prompt):]
    try:
        s = response.find("{"); e = response.rfind("}") + 1
        parsed = json.loads(response[s:e])
        cat = parsed.get("category","general_inquiry")
        if cat not in CATEGORIES: cat = "general_inquiry"
        return {"category":cat, "priority":parsed.get("priority","P4"),
                "summary":parsed.get("summary","Ticket requires review."),
                "suggested_action":parsed.get("suggested_action","Assign to agent."),
                "latency_ms":round(latency,1), "kb_articles":articles}
    except:
        return {"category":"general_inquiry","priority":"P4",
                "summary":"Could not parse response.",
                "suggested_action":"Assign to agent.",
                "latency_ms":round(latency,1),"kb_articles":articles}

app = FastAPI(title="AI Ticket Triage", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

@app.get("/health")
def health():
    return {"status":"healthy","model":MODEL_PATH}

@app.post("/triage", response_model=TriageResult)
def triage(req: TicketRequest):
    return infer(req.ticket_text, req.use_rag)
