# Multi-Domain RAG Chatbot Platform

A production-style Flask backend that serves multiple retrieval-augmented generation (RAG) chatbots — Finance policy Q&A, IT policy Q&A, and Enterprise Risk Management (ERM) analytics — behind SSO-protected APIs, with hybrid SQL + vector search routing and full audit logging.

> **Note:** This is a sanitized portfolio version of a system originally built for enterprise use. All company names, internal domains, credentials, and machine-specific file paths have been replaced with placeholders. The architecture, prompt design, and routing logic are unchanged.

---

## What it does

The platform exposes two authenticated endpoints, each backing a different chatbot experience:

| Endpoint | Purpose | Auth |
|---|---|---|
| `POST /get` | Finance and IT policy chatbots — answers strictly from a vector database of internal policy documents | Okta (production) |
| `POST /geterm` | ERM chatbot — answers questions about a risk register, blending structured SQL analytics with vector search | Okta (dev/service) |

Every conversation is logged to an Oracle database for audit and analytics purposes.

---

## Architecture

```
                        ┌───────────────────┐
                        │   Client / UI     │
                        └─────────┬─────────┘
                                  │  Bearer token
                     ┌────────────┴────────────┐
                     │     Flask API layer      │
                     │  (Talisman security,     │
                     │   CORS, Okta introspect) │
                     └───┬──────────┬──────────┘
                         │          │
             ┌───────────┘          └───────────┐
             ▼                                  ▼
     ┌───────────────┐                  ┌────────────────┐
     │ Finance / IT   │                  │   ERM Chatbot  │
     │ RAG chatbots   │                  │ (SQL + Vector) │
     └───────┬───────┘                  └───────┬────────┘
             │                                   │
     ┌───────▼────────┐                 ┌────────▼─────────┐
     │ Chroma vector   │                │ SQLite risk DB +  │
     │ stores (Azure   │                │ Chroma vector     │
     │ OpenAI embeds)  │                │ stores (ensemble  │
     └────────────────┘                 │ retriever by risk │
                                         │ level)            │
                                         └───────────────────┘
                         │
                         ▼
                ┌──────────────────┐
                │  Oracle DB        │
                │  (chat audit log) │
                └──────────────────┘
```

---

## Key engineering decisions

- **Hybrid routing for ERM queries.** Risk-register questions are first attempted through a SQL agent (`SQLDatabaseChain`) for precise, numeric answers. If the generated SQL touches free-text columns (descriptions, causes, impacts), returns no rows, or the query mentions themes, the system falls back to a vector-based `ConversationalRetrievalChain` instead — avoiding the classic RAG failure mode where structured questions get vague, hallucinated answers.
- **Keyword-driven retriever selection.** For the ERM bot, the query is scanned for severity language ("critical", "moderate", "minor", etc.) and routed to a matching Chroma collection (`High` / `Medium` / `Low` / `Immaterial`), then combined via an `EnsembleRetriever` across that collection's shards — keeping retrieval fast and relevant instead of searching the entire risk register for every query.
- **Grounding-first prompting.** All three chatbots use strict system prompts that forbid the model from answering outside the retrieved context, require explicit "not specified" fallbacks instead of inferred values, and strip out internal implementation details (e.g., "vector database") from user-facing responses.
- **Sharded vector collections.** Large document sets are split into multiple Chroma collections (`Finance_Batch1`–`8`, `chroma_High_part1`–`2`, etc.) and queried in a loop, which keeps each collection small enough to stay fast while still covering the full corpus.
- **Defense-in-depth on the API layer.** `flask-talisman` enforces HSTS, a strict CSP, `X-Frame-Options: DENY`, and forced HTTPS; CORS is locked to specific origins per route; every request is authenticated via Okta token introspection before touching any business logic.
- **Full audit trail.** Every question and answer, along with the user's identity and region/country (from the Okta token), is logged to Oracle for compliance and analytics.

---

## Tech stack

- **Backend:** Python, Flask, flask-talisman, flask-cors
- **LLM / RAG:** LangChain, Azure OpenAI (chat + embeddings), Chroma
- **Structured data:** SQLAlchemy, SQLite (risk register), Oracle (`oracledb`) for audit logging
- **Auth:** Okta (OAuth2 token introspection)

---

## Project structure

```
.
├── app.py              # Flask app, routes, auth decorators
├── .env.example        # Required environment variables (no real secrets)
├── requirements.txt    # Python dependencies
└── data/               # Local vector DB / SQLite DB storage (gitignored)
```

---

## Setup

1. **Clone and install dependencies**
   ```bash
   git clone https://github.com/<your-username>/<repo-name>.git
   cd <repo-name>
   pip install -r requirements.txt
   ```

2. **Configure environment variables**
   ```bash
   cp .env.example .env
   ```
   Fill in your own Azure OpenAI, Oracle, and Okta values in `.env`.

3. **Populate the vector stores**
   This repo expects pre-built Chroma collections at the paths defined by `FINANCE_DB_PATH`, `IT_DB_PATH`, and the ERM `chroma_data_prod/` directory. Point these at your own embedded document set.

4. **Run the app**
   ```bash
   python app.py
   ```
   The API will be available at `http://localhost:80`.

---

## API Reference

### `POST /get`
```json
{
  "chatbot": "finance",   // or "it"
  "msg": "What is the policy on inventory cycle counts?"
}
```
Returns a policy-grounded answer with source policy metadata.

### `POST /geterm`
```json
{
  "msg": "What are the top critical risks in the APAC region?"
}
```
Returns a risk-register-grounded answer with a follow-up disclaimer.

Both endpoints require `Authorization: Bearer <token>`.

---

## Disclaimer

This is a demonstration/portfolio build. It is not connected to any live company system, and all sample credentials, domains, and paths in this repo are placeholders.
