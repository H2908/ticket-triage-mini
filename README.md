# 🎫 AI Ticket Triage System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TinyLlama](https://img.shields.io/badge/TinyLlama-1.1B-FF6B6B?style=for-the-badge&logo=huggingface&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.33-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![ChromaDB](https://img.shields.io/badge/ChromaDB-RAG-7C3AED?style=for-the-badge)
![LoRA](https://img.shields.io/badge/LoRA-Fine--tuned-F59E0B?style=for-the-badge)

**An end-to-end AI pipeline that automatically classifies and routes customer support tickets using a fine-tuned LLM with RAG-augmented knowledge base retrieval.**

[Live Demo](#demo) • [Quick Start](#quick-start) • [Architecture](#architecture) • [API Docs](#api-reference)

</div>

---

## 📌 What It Does

Support teams at e-commerce companies receive hundreds of tickets daily with no automated way to prioritise or route them. This system solves that — it reads a raw support ticket, classifies it into one of **8 categories**, assigns a **priority level**, generates a **one-line summary**, and suggests the **exact next action** for the support agent.

All in under 100ms.

---

## ✨ Features

| Feature | Details |
|---|---|
| 🤖 Fine-tuned LLM | TinyLlama-1.1B trained with LoRA on 100 labelled support tickets |
| 🔍 RAG Pipeline | ChromaDB vector store with semantic search across 8 knowledge base articles |
| ⚡ REST API | FastAPI with Pydantic validation, auto-generated Swagger docs |
| 🎨 Live Demo | Streamlit frontend with sample tickets and KB article viewer |
| 📊 8 Categories | Full coverage of real e-commerce support scenarios |
| 🏷️ Priority Levels | P1 Critical → P4 Low with automated assignment |

---

## 🗂️ Ticket Categories

```
💳 Billing           →  P2 High      Invoice disputes, payment failures, subscription issues
📦 Order Management  →  P3 Medium    Tracking, cancellations, wrong items, delivery issues
↩️  Returns & Refunds →  P3 Medium    Return requests, refund status, exchange queries
🔐 Account Access    →  P2 High      Login failures, password resets, 2FA, account locks
🔧 Technical Support →  P2 High      App crashes, sync issues, API errors, bugs
💬 General Inquiry   →  P4 Low       Pricing, policies, shipping info, FAQs
🚨 Urgent Escalation →  P1 Critical  Fraud, data breaches, system outages, SLA breaches
⭐ Feedback          →  P4 Low       Compliments, complaints, feature requests
```

---

## 🏗️ Architecture

```
                    ┌─────────────────────────────────┐
                    │         Support Ticket           │
                    │   "I cannot log into my account" │
                    └──────────────┬──────────────────┘
                                   │
               ┌───────────────────┼───────────────────┐
               │                   │                   │
               ▼                   ▼                   │
    ┌─────────────────┐  ┌──────────────────┐         │
    │   ChromaDB      │  │   TinyLlama-1.1B │         │
    │   Vector Store  │  │   + LoRA Adapter │         │
    │                 │  │                  │         │
    │  8 KB articles  │  │  Fine-tuned on   │         │
    │  all-MiniLM-L6  │  │  100 tickets     │         │
    └────────┬────────┘  └────────┬─────────┘         │
             │                    │                    │
             │   RAG Context      │                    │
             └────────────────────┘                    │
                          │                            │
                          ▼                            │
              ┌───────────────────────┐                │
              │    FastAPI + Pydantic │                │
              │    /triage endpoint   │                │
              └───────────┬───────────┘                │
                          │                            │
                          ▼                            │
              ┌───────────────────────┐                │
              │  {                    │                │
              │    category: "account_│access"         │
              │    priority: "P2",    │                │
              │    summary: "...",    │                │
              │    action: "..."      │                │
              │  }                    │                │
              └───────────────────────┘                │
```

---

## 🚀 Quick Start

### 1. Clone the repo

```bash
git clone https://github.com/H2908/ticket-triage-mini.git
cd ticket-triage-mini
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the model

```bash
python train.py
# ~8 minutes on T4 GPU (Google Colab)
# Model saved to /content/model/final
```

### 4. Start the API

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
# Swagger UI → http://localhost:8000/docs
```

### 5. Launch the demo

```bash
streamlit run app.py
# Demo → http://localhost:8501
```

---

## 📡 API Reference

### `POST /triage`

Classify a support ticket.

**Request:**
```json
{
  "ticket_text": "I was charged twice for my last order. Please refund.",
  "use_rag": true
}
```

**Response:**
```json
{
  "category": "billing",
  "priority": "P2",
  "summary": "Customer has a billing or payment issue.",
  "suggested_action": "Escalate to billing team within 24 hours.",
  "latency_ms": 43.2,
  "kb_articles": [
    {
      "title": "Billing & Payments",
      "content": "Invoices generated on 1st of each month...",
      "relevance": 0.912
    }
  ]
}
```

### `GET /health`

```json
{ "status": "healthy", "model": "/content/model/final" }
```

---

## 📁 Project Structure

```
ticket-triage-mini/
│
├── 📄 data.json          # 100 hand-labelled support tickets (8 categories)
├── 🏋️ train.py           # TinyLlama LoRA fine-tuning pipeline
├── 🔍 rag.py             # ChromaDB vector store + semantic retrieval
├── ⚡ api.py             # FastAPI + Pydantic REST API
├── 🎨 app.py             # Streamlit live demo
└── 📦 requirements.txt   # Python dependencies
```

---

## 🧠 Model Details

| Setting | Value |
|---|---|
| Base model | TinyLlama-1.1B-Chat-v1.0 |
| Fine-tuning method | LoRA (r=8, alpha=16) |
| Target modules | q_proj, v_proj |
| Quantisation | 4-bit NF4 (QLoRA) |
| Training epochs | 5 |
| Batch size | 4 |
| Learning rate | 2e-4 |
| Max sequence length | 128 tokens |
| Training data | 100 tickets → 80 train / 20 val |
| Training time | ~8 minutes on T4 GPU |

---

## 🔍 RAG Pipeline

The system retrieves the top-2 most semantically relevant knowledge base articles for each ticket before passing context to the model. This grounds the model's output in verified support documentation and reduces hallucinations.

```python
# Example retrieval
from rag import retrieve

articles = retrieve("I cannot log into my account")
# Returns:
# [{"title": "Account Access & Security", "relevance": 0.94, ...},
#  {"title": "General Information",       "relevance": 0.71, ...}]
```

Embeddings generated using `all-MiniLM-L6-v2` via sentence-transformers.
Vector similarity computed using cosine distance in ChromaDB.

---

## 💻 Tech Stack

```
Language Model    TinyLlama-1.1B-Chat-v1.0
Fine-tuning       LoRA via PEFT + HuggingFace Transformers
Quantisation      BitsAndBytes 4-bit NF4
Vector Database   ChromaDB
Embeddings        sentence-transformers (all-MiniLM-L6-v2)
API Framework     FastAPI + Pydantic v2
Frontend          Streamlit
Training          HuggingFace Trainer
```

---

## 🎯 Example Predictions

| Ticket | Category | Priority |
|---|---|---|
| "I was charged twice for my order" | billing | P2 🟠 |
| "URGENT: System down, losing £1000/min" | urgent_escalation | P1 🔴 |
| "Cannot log in — password rejected" | account_access | P2 🟠 |
| "App crashes on iPhone 14 settings" | technical_support | P2 🟠 |
| "Your support team was fantastic!" | feedback | P4 🟢 |
| "Do you offer a student discount?" | general_inquiry | P4 🟢 |

---

## 🗺️ Roadmap

- [ ] Push trained model to HuggingFace Hub
- [ ] Deploy live demo to HuggingFace Spaces
- [ ] Add MLflow experiment tracking
- [ ] Add Prometheus metrics endpoint
- [ ] Expand dataset to 500+ tickets
- [ ] Add confidence score to predictions
- [ ] Docker + docker-compose deployment

---

## 👨‍💻 Built By

**Harshit Taneja**
MSc Computer Science — University of Birmingham, UK

[![GitHub](https://img.shields.io/badge/GitHub-H2908-181717?style=flat&logo=github)](https://github.com/H2908)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-harshit--taneja-0A66C2?style=flat&logo=linkedin)](https://linkedin.com/in/harshit-taneja-921821223)
[![Email](https://img.shields.io/badge/Email-tanejaharshit79@gmail.com-EA4335?style=flat&logo=gmail)](mailto:tanejaharshit79@gmail.com)

---

<div align="center">
  <sub>If you found this useful, please ⭐ star the repo!</sub>
</div>
